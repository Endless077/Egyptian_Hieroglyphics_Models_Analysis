# Libraries
import os
import logging
from typing import List, Tuple, Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

import timm
import hydra
import seaborn as sns
import matplotlib.pyplot as plt

from omegaconf import DictConfig
from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Configure logging to output information during training
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the transformations applied to the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.RandomRotation(30),  
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),  
        transforms.CenterCrop(224),  
        transforms.ToTensor(),  
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  
    ]),
}


def plot_and_save_metrics(
    train_losses: List[float], 
    val_losses: List[float], 
    train_accuracies: List[float], 
    val_accuracies: List[float], 
    save_dir: str, 
    model_num: str
    ) -> None:
    """
    Plots and saves learning curves for training and validation losses and accuracies.

    Args:
        train_losses (List[float]): List of training losses for each epoch.
        val_losses (List[float]): List of validation losses for each epoch.
        train_accuracies (List[float]): List of training accuracies for each epoch.
        val_accuracies (List[float]): List of validation accuracies for each epoch.
        save_dir (str): Directory where the plots will be saved.
        model_num (str): Identifier for the model to include in the filename.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'go-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Save the plot
    plt.savefig(os.path.join(save_dir, f'learning_curves_model_{model_num}.png'))
    plt.show()


def train_model(
    model: nn.Module, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    device: torch.device, 
    scheduler: torch.optim.lr_scheduler, 
    num_epochs: int = 25, 
    early_stopping_patience: int = 10
    ) -> tuple:
    """
    Trains the model with early stopping and learning rate scheduling.

    Args:
        model (nn.Module): The model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        device (torch.device): Device to run the model on (CPU or GPU).
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model.
        early_stopping_patience (int): Number of epochs to wait for improvement before stopping.

    Returns:
        Tuple[float, Dict[str, torch.Tensor], List[float], List[float], List[float], List[float]]:
            - best_val_acc (float): Best validation accuracy achieved.
            - best_model_weights (dict): Weights of the best model.
            - train_losses (list): List of training losses for each epoch.
            - val_losses (list): List of validation losses for each epoch.
            - train_accuracies (list): List of training accuracies for each epoch.
            - val_accuracies (list): List of validation accuracies for each epoch.
    """
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    best_model_weights = None

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []

    logging.info("Starting model training...")

    for epoch in range(num_epochs):
        model.train()
        running_loss, corrects = 0.0, 0
        
        logging.info(f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels.data).sum().item()

            if batch_idx % 10 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = corrects / len(train_loader.dataset)
        logging.info(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}')

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation phase
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_weights = model.state_dict()
            logging.info("Validation accuracy improved, saving model weights...")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break

    model.load_state_dict(best_model_weights)
    return best_val_acc, best_model_weights, train_losses, val_losses, train_accuracies, val_accuracies


def validate_model(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
        ) -> Tuple[float, float]:
    """
    Validates the model on the validation dataset.

    Args:
        model (nn.Module): The model to validate.
        val_loader (DataLoader): DataLoader for the validation dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        Tuple[float, float]: 
            - val_loss (float): Validation loss.
            - val_acc (float): Validation accuracy.
    """
    model.eval()
    val_loss, corrects = 0.0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels.data).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = corrects / len(val_loader.dataset)
    logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')
    
    return val_loss, val_acc


def test(
    model: nn.Module, 
    test_loader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
    ) -> Tuple[float, np.ndarray, float, float, float]:
    """
    Tests the model and computes accuracy metrics.

    Args:
        model (nn.Module): The model to test.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): Device to run the model on (CPU or GPU).

    Returns:
        Tuple[float, np.ndarray, float, float, float]: 
            - accuracy (float): Test accuracy.
            - confusion_mat (np.ndarray): Confusion matrix.
            - precision (float): Precision score.
            - recall (float): Recall score.
            - f1 (float): F1 score.
    """
    model.eval()
    corrects, total_loss = 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            corrects += (preds == labels.data).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = corrects / len(test_loader.dataset)
    average_loss = total_loss / len(test_loader.dataset)
    confusion_mat = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    logging.info(f'Test Loss: {average_loss:.4f}, Test Accuracy: {accuracy:.4f}')
    return accuracy, confusion_mat, precision, recall, f1


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to set up training and testing pipeline.

    Args:
        cfg (DictConfig): Configuration object containing paths and hyperparameters.
    """
    tracker = EmissionsTracker()
    tracker.start()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    logging.info("Loading dataset...")
    train_dataset = datasets.ImageFolder(root=cfg.dataset.train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=cfg.dataset.val_path, transform=data_transforms['val'])
    test_dataset = datasets.ImageFolder(root=cfg.dataset.test_path, transform=data_transforms['val'])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, shuffle=False)

    # Load the model using timm
    logging.info("Loading model...")
    model = timm.create_model(cfg.model.name, pretrained=True, num_classes=len(train_dataset.classes))
    model.to(device)

    # Define the loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=cfg.training.patience)

    # Train the model
    best_val_acc, best_model_weights, train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, criterion, optimizer, train_loader, val_loader, device, scheduler, 
        num_epochs=cfg.training.num_epochs, early_stopping_patience=cfg.training.early_stopping_patience
    )

    # Save model weights
    save_dir = os.path.join(cfg.results.dir, 'model_weights')
    os.makedirs(save_dir, exist_ok=True)
    torch.save(best_model_weights, os.path.join(save_dir, 'best_model.pth'))

    # Plot and save metrics
    plot_and_save_metrics(train_losses, val_losses, train_accuracies, val_accuracies, save_dir, cfg.model.name)

    # Test the model
    accuracy, confusion_mat, precision, recall, f1 = test(model, test_loader, criterion, device)
    
    logging.info(f'Test Results - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))
    
    tracker.stop()


if __name__ == "__main__":
    main()
