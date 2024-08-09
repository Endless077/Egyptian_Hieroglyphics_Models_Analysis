
import os
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms, datasets
import hydra
from omegaconf import DictConfig

from ATCNet import ATCNet
from model import Glyphnet

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
    model.to(device)  # Move model to the device
    best_val_acc = 0.0
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        corrects = 0
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to the device
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / len(val_loader.dataset)
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch + 1} training time: {epoch_time:.2f} seconds')

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print("Validation accuracy improved, saving model...")
            torch.save(model.state_dict(), 'checkpoint.pt')

    return best_val_acc


# Funzione di predizione per l'ensemble
def predict_ensemble(models, data_loader, device):
    all_predictions = []
    for model in models:
        model.to(device)  # Move model to the device
        model.eval()
        model_predictions = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)  # Move data to the device
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                model_predictions.append(preds.cpu().numpy())
        all_predictions.append(np.concatenate(model_predictions))
    return np.array(all_predictions)


# Funzione per combinare le predizioni (voting)
def combine_predictions(predictions):
    combined_predictions = []
    for i in range(predictions.shape[1]):
        votes = np.bincount(predictions[:, i])
        combined_predictions.append(np.argmax(votes))
    return np.array(combined_predictions)


@hydra.main(config_path="./configs", config_name="config_ensamble_learning")
def main(cfg: DictConfig):
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    val_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.val_path)

    # Caricamento del dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])

    def create_data_loader(dataset, batch_size, indices):
        sampler = SubsetRandomSampler(indices)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    num_samples = 5  # Numero di campioni per il bagging
    sample_size = len(train_dataset) // num_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Addestramento di modelli ATCNet
    atc_models = []
    for i in range(num_samples):
        print(f"Training ATCNet model {i + 1}/{num_samples}")
        sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)
        train_loader = create_data_loader(train_dataset, cfg.train.batch_size, sample_indices)
        val_loader = create_data_loader(val_dataset, cfg.train.batch_size, list(range(len(val_dataset))))
        model = ATCNet(n_classes=len(train_dataset.classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                    num_epochs=cfg.train.num_epochs)
        atc_models.append(model)

    # Addestramento di modelli GlyphoNet
    glypho_models = []
    for i in range(num_samples):
        print(f"Training GlyphoNet model {i + 1}/{num_samples}")
        train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
        test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)
        if test_path:
            test_path = os.path.join(hydra.utils.get_original_cwd(), test_path)
        train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in os.listdir(train_path)]))}
        test_labels_set = {l for l in [p.strip("/") for p in os.listdir(test_path)]}
        test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

        sample_indices = np.random.choice(len(train_dataset), sample_size, replace=False)
        train_loader = create_data_loader(train_dataset, cfg.train.batch_size, sample_indices)
        val_loader = create_data_loader(val_dataset, cfg.train.batch_size, list(range(len(val_dataset))))
        model = Glyphnet(num_classes=len(train_labels)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)
        train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                    num_epochs=cfg.train.num_epochs)
        glypho_models.append(model)

    # Combina le predizioni dei modelli ATCNet e GlyphoNet
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    atc_predictions = predict_ensemble(atc_models, val_loader, device)
    glypho_predictions = predict_ensemble(glypho_models, val_loader, device)

    # Combinazione delle predizioni (voting)
    combined_predictions = combine_predictions(np.concatenate((atc_predictions, glypho_predictions), axis=0))
    true_labels = [label for _, label in val_loader.dataset]

    # Calcola l'accuratezza finale
    final_accuracy = accuracy_score(true_labels, combined_predictions)
    print(f'Final ensemble accuracy: {final_accuracy:.4f}')


if __name__ == "__main__":
    main()
