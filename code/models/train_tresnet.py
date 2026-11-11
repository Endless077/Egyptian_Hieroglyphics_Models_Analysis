# Libraries
import os
import time
import copy
import timm
import hydra
import logging
from codecarbon import EmissionsTracker

import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Configuring the logger
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Function to load the dataset
def setup_data_loaders(train_path: str, valid_path: str, batch_size: int, img_size: int) -> tuple:
    """
    Configures the DataLoaders for training and validation.

    Args:
        train_path (str): Path to the training dataset.
        valid_path (str): Path to the validation dataset.
        batch_size (int): Batch size.
        img_size (int): Size of the image.

    Returns:
        tuple: (DataLoaders for training and validation, sizes of the datasets, class names)
    """
    # Data transformations for training and validation datasets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(path), data_transforms[x]) for x, path in zip(['train', 'valid'], [train_path, valid_path])}
    
    # Create DataLoaders for training and validation
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=(x == 'train'), num_workers=4) for x in ['train', 'valid']}
    
    # Get the sizes of the datasets
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    return dataloaders, dataset_sizes, image_datasets['train'].classes


# Model training function
def train_model(model: nn.Module,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                scheduler: optim.lr_scheduler,
                dataloaders: dict,
                dataset_sizes: dict,
                device: torch.device,
                num_epochs: int) -> tuple:
    """
    Trains the model and saves the best weights.

    Args:
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        dataloaders (dict): DataLoaders for training and validation.
        dataset_sizes (dict): Sizes of the datasets.
        device (torch.device): Device to train the model on.
        num_epochs (int): Number of training epochs.

    Returns:
        tuple: (model with best weights, training losses, validation losses, training accuracies, validation accuracies)
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses, valid_losses = [], []
    train_accuracies, valid_accuracies = [], []

    for epoch in range(num_epochs):
        log.info(f'Epoch {epoch}/{num_epochs - 1}')
        log.info('-' * 10)

        for phase in ['train', 'valid']:
            # Set model to training or evaluation mode
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over the data
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Forward pass
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass and optimization only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # Step the scheduler
            if phase == 'train':
                scheduler.step()

            # Calculate epoch loss and accuracy
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            log.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Store metrics
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                valid_losses.append(epoch_loss)
                valid_accuracies.append(epoch_acc.item())

            # Save best model weights
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    log.info(f'Training complete in {time.time() - since:.0f}s. Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies


# Function to train with specific parameters
def train_with_params(cfg: DictConfig, learning_rate: float, batch_size: int) -> tuple:
    """
    Configures and trains the model with specific parameters.

    Args:
        cfg (DictConfig): Hydra configuration.
        learning_rate (float): Learning rate to use.
        batch_size (int): Batch size.

    Returns:
        tuple: (trained model, configured optimizer, configured scheduler, dataloaders, sizes of the datasets, class names, training losses, validation losses, training accuracies, validation accuracies)
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    valid_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.valid_path)

    # Setup data loaders
    dataloaders, dataset_sizes, class_names = setup_data_loaders(train_path, valid_path, batch_size, cfg.train.img_size)

    # Initialize the model
    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained, num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler_step_size, gamma=cfg.train.scheduler_gamma)

    return train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, cfg.train.num_epochs)


# Function to evaluate the model and calculate metrics
def evaluate_model(model: nn.Module, dataloaders: dict, dataset_sizes: dict, device: torch.device) -> tuple:
    """
    Evaluates the model and calculates precision, recall, and F1 metrics.

    Args:
        model (torch.nn.Module): Model to evaluate.
        dataloaders (dict): DataLoaders for validation.
        dataset_sizes (dict): Sizes of the datasets.
        device (torch.device): Device to run evaluation on.

    Returns:
        tuple: (accuracy, precision, recall, F1 score, confusion matrix)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy and metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm


# Function to plot the confusion matrix
def plot_confusion_matrix(cm: np.ndarray, class_names: list) -> None:
    """
    Plots the confusion matrix using seaborn.

    Args:
        cm (numpy.ndarray): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()


@hydra.main(config_path='config', config_name='config')
def main(cfg: DictConfig):
    """
    Main function to run the model training and evaluation.

    Args:
        cfg (DictConfig): Hydra configuration.
    """
    tracker = EmissionsTracker()
    tracker.start()

    learning_rate = cfg.train.learning_rate
    batch_size = cfg.train.batch_size

    # Train the model with specified parameters
    model, optimizer, scheduler, dataloaders, dataset_sizes, class_names, train_losses, valid_losses, train_accuracies, valid_accuracies = train_with_params(cfg, learning_rate, batch_size)

    # Evaluate the trained model
    accuracy, precision, recall, f1, cm = evaluate_model(model, dataloaders, dataset_sizes, cfg.device)

    # Log evaluation metrics
    log.info(f'Accuracy: {accuracy:.4f}')
    log.info(f'Precision: {precision:.4f}')
    log.info(f'Recall: {recall:.4f}')
    log.info(f'F1 Score: {f1:.4f}')

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names)

    tracker.stop()


if __name__ == "__main__":
    main()
