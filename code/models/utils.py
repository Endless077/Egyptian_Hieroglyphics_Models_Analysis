# Libraries
import os
import logging
import numpy as np
import seaborn as sns
from os import listdir
import matplotlib.pyplot as plt

import torch
import hydra
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, log_loss)

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from models.GlyphNet.model import Glyphnet

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define a grid of hyperparameters for search
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128]
}

# Transformations for training data
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def load_and_split_data(cfg: dict) -> tuple[dict]:
    """
    Load training and testing data and create a label mapping for training data.

    Args:
        cfg (DictConfig): Hydra configuration containing data paths.

    Returns:
        Tuple: Mapping of training labels, test data path, training data path.
    """
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    # Create a mapping of training labels
    train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in listdir(train_path)]))}
    return train_labels, test_path, train_path


def move_to_device(data: torch.Tensor, target: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Move data and target tensors to the specified device.

    Args:
        data (torch.Tensor): The input data tensor.
        target (torch.Tensor): The target tensor (labels).
        device (torch.device): The device to move the tensors to (CPU or GPU).

    Returns:
        tuple: A tuple containing the data and target tensors on the specified device.
    """
    return data.to(device), target.to(device)


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, num_classes: int) -> None:
    """
    Evaluate the model on a test set and compute evaluation metrics.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader containing the test data.
        device (torch.device): The device to run the model on (CPU or GPU).
        num_classes (int): Number of classes in the dataset.

    Returns:
        None
    """
    # Set the model to evaluation mode
    model.eval()
    all_predictions, all_gold, all_logits = [], [], []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = move_to_device(data, target, device)         # Move to device
            output = model(data)                                        # Get model predictions
            pred = softmax2predictions(output)                          # Convert output to predictions
            all_predictions.append(pred.cpu().numpy())                  # Store predictions
            all_gold.append(target.cpu().numpy())                       # Store true labels
            all_logits.append(F.softmax(output, dim=1).cpu().numpy())   # Store logits 

    # Combine predictions
    y_pred = np.concatenate(all_predictions)
    # Combine true labels
    y_true = np.concatenate(all_gold)
    # Combine logits
    y_logits = np.concatenate(all_logits)

    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    try:
        roc_auc = roc_auc_score(y_true, y_logits, multi_class="ovr", labels=np.arange(num_classes))
    except ValueError as e:
        roc_auc = None
        logging.warning(f"ROC-AUC could not be computed: {e}")

    logloss = log_loss(y_true, y_logits, labels=np.arange(num_classes))

    # Log metrics
    logging.info(f"    Acc.: {acc * 100:.2f}%; Precision: {precision * 100:.2f}%; "
                 f"Recall: {recall * 100:.2f}%; F1: {f1 * 100:.2f}%")
    if roc_auc is not None:
        logging.info(f"    ROC-AUC: {roc_auc:.4f}")
    logging.info(f"    Log Loss: {logloss:.4f}")

    # Plot confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def train(model: nn.Module,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          loss_function: nn.Module,
          epoch: int,
          device: torch.device,
          batch_reports_interval: int = 100) -> None:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader containing the training data.
        optimizer (optim.Optimizer): Optimizer to update model weights.
        loss_function (nn.Module): Loss function to use.
        epoch (int): Current epoch number.
        device (torch.device): The device to run the model on (CPU or GPU).
        batch_reports_interval (int): Number of batches after which to log status.

    Returns:
        None
    """
    # Set the model to training mode
    model.train()
    # Accumulator for loss
    loss_accum = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = move_to_device(data, target, device)     # Move to device
        optimizer.zero_grad()                                   # Reset gradients
        output = model(data)                                    # Get predictions
        loss = loss_function(output, target)                    # Compute loss
        loss_accum += loss.item() / len(data)                   # Save loss for statistics
        loss.backward()                                         # Compute gradients
        optimizer.step()                                        # Update model weights

        if batch_idx % batch_reports_interval == 0:
            logging.info(f'Train Epoch: {epoch + 1} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                         f'({100. * batch_idx / len(train_loader):.0f}%)]\tAveraged Epoch Loss: {loss_accum / (batch_idx + 1):.6f}')


def softmax2predictions(output: torch.Tensor) -> torch.Tensor:
    """
    Convert the model's softmax output to class predictions.

    Args:
        output (torch.Tensor): Output from the model.

    Returns:
        torch.Tensor: Indices of predicted classes.
    """
    return torch.topk(output, k=1, dim=-1).indices.flatten()


def validate(model: nn.Module, val_loader: DataLoader, loss_function: nn.Module, device) -> tuple:
    """
    Validate the model on the validation set and compute metrics.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader containing validation data.
        loss_function (nn.Module): Loss function to use.
        device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Average loss over validation set, true labels, predictions.
    """
    # Set the model to evaluation mode
    model.eval()
    val_loss = 0.0
    all_predictions, all_gold = [], []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = move_to_device(data, target, device)     # Move to device
            output = model(data)                                    # Get predictions
            pred = softmax2predictions(output)                      # Convert output to predictions

            val_loss += loss_function(output, target).sum().item()  # Accumulate validation loss
            all_predictions.append(pred.cpu().numpy())              # Store predictions
            all_gold.append(target.cpu().numpy())                   # Store true labels

    # Calculate average validation loss
    val_loss /= len(val_loader.dataset)

    y_pred = np.concatenate(all_predictions)    # Combine predictions
    y_true = np.concatenate(all_gold)           # Combine true labels

    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)
    logging.info(f'    Validation loss: {val_loss:.4f}, Accuracy: {np.sum(y_pred == y_true)}/{len(val_loader.dataset)} '
                 f'({100. * acc:.0f}%)')

    # Return loss, true labels, and predictions
    return val_loss, y_true, y_pred


def save_model(model: nn.Module, model_name: str) -> None:
    """Save the model state to a specified file.

    Args:
        model (nn.Module): The model to be saved.
        model_name (str): The filename where the model will be saved.

    Raises:
        Exception: If there is an error during saving the model.
    """
    torch.save(model.state_dict(), model_name)
    logging.info(f"Model saved to {model_name}")


def train_and_evaluate(train_loader: DataLoader,
                       val_loader: DataLoader,
                       model: nn.Module,
                       optimizer: optim.Optimizer,
                       scheduler: optim.lr_scheduler._LRScheduler,
                       loss_function: nn.Module,
                       device: torch.device,
                       epochs: int,
                       model_name: str,
                       early_stopping_patience: int = 10) -> None:
    """
    Train and evaluate the model using an early stopping mechanism.

    Args:
        train_loader (DataLoader): DataLoader containing training data.
        val_loader (DataLoader): DataLoader containing validation data.
        model (nn.Module): The model to train and evaluate.
        optimizer (optim.Optimizer): Optimizer to update model weights.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler to update learning rate.
        loss_function (nn.Module): Loss function to use.
        device (torch.device): The device to run the model on (CPU or GPU).
        epochs (int): Number of epochs to train for.
        model_name (str): Name of the model for saving purposes.
        early_stopping_patience (int): Number of epochs to wait before stopping if no improvement.

    Returns:
        None
    """
    # Initialize best validation loss
    best_val_loss = float("inf")

    # Counter for epochs with no improvement
    no_improvement_epochs = 0

    for epoch in range(epochs):
        logging.info(f"Epoch {epoch + 1}/{epochs}:")

        # Train model
        train(model, train_loader, optimizer, loss_function, epoch, device)

        # Validate model
        val_loss, _, _ = validate(model, val_loader, loss_function, device)

        # Check if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss        # Update best validation loss
            no_improvement_epochs = 0       # Reset counter
            save_model(model, model_name)   # Save model state
        else:
            no_improvement_epochs += 1  # Increment counter
            if no_improvement_epochs >= early_stopping_patience:
                logging.info("Early stopping triggered.")
                # Stop training if no improvement
                break


def run_training(cfg: dict) -> None:
    """
    Main function to run the training process.

    Args:
        cfg (DictConfig): Hydra configuration containing all settings.

    Returns:
        None
    """
    # Set device for training (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and split data
    train_labels, test_path, train_path = load_and_split_data(cfg)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)             # Load training dataset
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True)  # Create DataLoader for training
    val_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=False)   # Create DataLoader for validation

    # Initialize model, loss function, optimizer
    model = Glyphnet(num_classes=len(train_labels)).to(device)              # Create model instance
    loss_function = nn.CrossEntropyLoss()                                   # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)  # Initialize optimizer

    # Run training and evaluation
    train_and_evaluate(train_loader, val_loader, model, optimizer, None, loss_function, device, cfg.train.epochs,
                       model_name=os.path.join(hydra.utils.get_original_cwd(), "model.pth"),
                       early_stopping_patience=cfg.train.early_stopping_patience)

    logging.info("Training completed.")
