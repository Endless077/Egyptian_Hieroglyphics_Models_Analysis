import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import hydra
from omegaconf import DictConfig

from ATCNet import ATCNet

# Definizione delle trasformazioni per il dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ]),
}


# Funzione di addestramento
def train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler, num_epochs=25):
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()

        # Addestramento
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        # Validazione
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
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch + 1} training time: {epoch_time:.2f} seconds')

        scheduler.step(val_loss)  # Step del scheduler

        # Salva il modello se la validazione migliora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Validation accuracy improved, saving model...")

    return best_val_acc


@hydra.main(config_path="./configs", config_name="config_transfer_learning")
def main(cfg: DictConfig):
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    val_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.val_path)

    # Caricamento del dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])

    # Creazione del DataLoader
    def create_data_loader(dataset, batch_size):
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Creazione dei DataLoader
    train_loader = create_data_loader(train_dataset, cfg.train.batch_size)
    val_loader = create_data_loader(val_dataset, cfg.train.batch_size)

    # Definizione del modello
    model = ATCNet(n_classes=len(train_dataset.classes))

    # Impostazione del dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definizione della loss function e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Scheduler per il learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Addestramento del modello
    train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                num_epochs=cfg.train.num_epochs)


if __name__ == "__main__":
    main()
