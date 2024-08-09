import os
import time
import torch
from sklearn.model_selection import ParameterGrid
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import hydra
from omegaconf import DictConfig
from efficientnet_pytorch import EfficientNet

# Definizione delle trasformazioni per il dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
}


# Funzione per modificare il modello EfficientNet per immagini in scala di grigi
def modify_efficientnet(model):
    model._conv_stem = nn.Conv2d(1, model._conv_stem.out_channels,
                                 kernel_size=model._conv_stem.kernel_size,
                                 stride=model._conv_stem.stride,
                                 padding=model._conv_stem.padding,
                                 bias=False)
    return model


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
    train_dataset = ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = ImageFolder(root=val_path, transform=data_transforms['val'])

    # Creazione del DataLoader
    def create_data_loader(batch_size):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Definizione della grid search
    param_grid = {
        'batch_size': [16, 32, 64],
        'learning_rate': [0.1, 0.01, 0.001],
        'dropout': [0.3, 0.5, 0.7],
        'num_epochs': [25]
    }

    grid = ParameterGrid(param_grid)
    best_params = None
    best_val_acc = 0.0

    # Loop attraverso tutte le combinazioni di iperparametri
    for params in grid:
        print(f"Testing combination: {params}")
        train_loader, val_loader = create_data_loader(params['batch_size'])

        # Caricamento del modello pre-addestrato
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model = modify_efficientnet(model)  # Modifica per immagini in scala di grigi

        # Congelare tutti i layer tranne l'ultimo
        for param in model.parameters():
            param.requires_grad = False

        # Scongelare solo l'ultimo strato
        model._fc.requires_grad = True

        # Modifica dell'ultimo strato
        num_features = model._fc.in_features
        model._fc = nn.Sequential(
            nn.Dropout(params['dropout']),  # Aggiunta di Dropout per regolarizzazione
            nn.Linear(num_features, len(train_dataset.classes))
        )

        # Impostazione del dispositivo
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Definizione della loss function e dell'ottimizzatore
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params['learning_rate'])

        # Scheduler per il learning rate
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

        # Addestramento del modello
        val_acc = train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                              num_epochs=params['num_epochs'])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = params

    print(f"Best params: {best_params}, Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
