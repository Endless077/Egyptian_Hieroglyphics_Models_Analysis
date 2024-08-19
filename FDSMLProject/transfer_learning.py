import logging
import os
import time
import torch
from codecarbon import EmissionsTracker
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import hydra
from omegaconf import DictConfig
import pretrainedmodels
import torch.nn.utils

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the transformations for the dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Apply ToTensor before Normalize
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize after ToTensor
        transforms.RandomErasing(p=0.5),  # Optional: RandomErasing after Normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Apply ToTensor before Normalize
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize after ToTensor
    ]),
}


# Function to load the Xception model without pre-trained weights
def load_custom_xception_model(pretrained=False):
    model = pretrainedmodels.__dict__['xception'](pretrained=True)
    logging.info("Xception model loaded with pre-trained weights.")
    return model


# Function to modify the model for grayscale images
def modify_model(model, model_name):
    logging.info(f"Modifying the {model_name} model for grayscale images.")
    if model_name == 'resnet50':
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=False)
        model.bn1 = nn.BatchNorm2d(model.conv1.out_channels)  # Add BatchNorm
    elif model_name == 'inception_v3':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, model.Conv2d_1a_3x3.conv.out_channels,
                                             kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
                                             stride=model.Conv2d_1a_3x3.conv.stride,
                                             padding=model.Conv2d_1a_3x3.conv.padding,
                                             bias=False)
        model.Conv2d_1a_3x3.bn = nn.BatchNorm2d(model.Conv2d_1a_3x3.conv.out_channels)  # Add BatchNorm
    elif model_name == 'xception':
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=False)
        model.bn1 = nn.BatchNorm2d(model.conv1.out_channels)  # Add BatchNorm
    logging.info(f"{model_name} model modification completed.")
    return model


# Training function

# Experiment with a lower max_norm for gradient clipping
max_norm_value = 0.1


def train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler, num_epochs=25,
                model_name="model"):
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}.")

        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping with a lower max_norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 10 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch + 1} completed. Training Loss: {epoch_loss:.4f}')

        # Validation
        model.eval()
        val_loss = 0.0
        corrects = 0
        all_predictions, all_gold = [], []
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

            all_predictions.append(preds.cpu().numpy())
            all_gold.append(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        logging.info(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        end_time = time.time()
        epoch_time = end_time - start_time
        logging.info(f'Epoch {epoch + 1} training time: {epoch_time:.2f} seconds')

        scheduler.step(val_loss)  # Step the scheduler

        # Save the model if validation improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, f'best_model_{model_name}.pth')
            logging.info(f"Validation accuracy improved to {val_acc:.4f}. Saving model weights...")

    logging.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc, best_model_weights


@hydra.main(config_path="./configs", config_name="config_transfer_learning")
def main(cfg: DictConfig):
    logging.info("Starting transfer learning training process.")

    # Inizializzazione di CodeCarbon
    tracker = EmissionsTracker()
    tracker.start()

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    val_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.val_path)

    # Load the dataset
    logging.info(f"Loading dataset from {train_path} and {val_path}.")
    train_dataset = ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = ImageFolder(root=val_path, transform=data_transforms['val'])

    # Create DataLoader
    def create_data_loader(batch_size):
        logging.info(f"Creating DataLoader with batch size: {batch_size}.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Set the hyperparameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 25

    logging.info(f"Using batch size: {batch_size}, learning rate: {learning_rate}, number of epochs: {num_epochs}")

    train_loader, val_loader = create_data_loader(batch_size)

    # Load model with pre-trained weights
    if cfg.model.name == 'resnet50':
        logging.info("Loading ResNet50 model with pre-trained weights.")
        model = models.resnet50(pretrained=True)
    elif cfg.model.name == 'inception_v3':
        logging.info("Loading Inception V3 model with pre-trained weights.")
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif cfg.model.name == 'xception':
        logging.info("Loading Xception model with pre-trained weights.")
        model = load_custom_xception_model(pretrained=True)

    model = modify_model(model, cfg.model.name)  # Modify for grayscale images

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Added weight decay

    # Scheduler for learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Train the model
    val_acc, _ = train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                             num_epochs=num_epochs)
    logging.info(f"Final validation accuracy: {val_acc:.4f}")

    # Interrompi il tracker e salva le metriche di emissione
    emissions = tracker.stop()
    logging.info(f"Emissioni di CO2 generate durante l'addestramento: {emissions} kg")


if __name__ == "__main__":
    main()
