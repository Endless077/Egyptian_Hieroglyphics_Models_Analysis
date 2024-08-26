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

# Definizione delle trasformazioni per il dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # Applica ToTensor prima di Normalize
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalizza dopo ToTensor
        transforms.RandomErasing(p=0.5),  # Opzionale: RandomErasing dopo Normalize
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Applica ToTensor prima di Normalize
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalizza dopo ToTensor
    ]),
}

# Funzione per caricare il modello Xception senza pesi pre-addestrati
def load_custom_xception_model(pretrained=False):
    """
    Carica il modello Xception con o senza pesi pre-addestrati.

    :param pretrained: Booleano, se True carica i pesi pre-addestrati.
    :return: Modello Xception.
    """
    model = pretrainedmodels.__dict__['xception'](pretrained=True)
    logging.info("Modello Xception caricato con pesi pre-addestrati.")
    return model

# Funzione per modificare il modello per immagini in scala di grigi
def modify_model(model, model_name):
    """
    Modifica il modello per accettare immagini in scala di grigi invece di RGB.

    :param model: Modello PyTorch.
    :param model_name: Nome del modello.
    :return: Modello modificato.
    """
    logging.info(f"Modifica del modello {model_name} per immagini in scala di grigi.")
    if model_name == 'resnet50':
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=False)
        model.bn1 = nn.BatchNorm2d(model.conv1.out_channels)  # Aggiungi BatchNorm
    elif model_name == 'inception_v3':
        model.Conv2d_1a_3x3.conv = nn.Conv2d(3, model.Conv2d_1a_3x3.conv.out_channels,
                                             kernel_size=model.Conv2d_1a_3x3.conv.kernel_size,
                                             stride=model.Conv2d_1a_3x3.conv.stride,
                                             padding=model.Conv2d_1a_3x3.conv.padding,
                                             bias=False)
        model.Conv2d_1a_3x3.bn = nn.BatchNorm2d(model.Conv2d_1a_3x3.conv.out_channels)  # Aggiungi BatchNorm
    elif model_name == 'xception':
        model.conv1 = nn.Conv2d(3, model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=False)
        model.bn1 = nn.BatchNorm2d(model.conv1.out_channels)  # Aggiungi BatchNorm
    logging.info(f"Modifica del modello {model_name} completata.")
    return model

# Valore massimo per il norm del gradiente per il gradient clipping
max_norm_value = 0.1

# Funzione di addestramento del modello
def train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler, num_epochs=25,
                model_name="model"):
    """
    Addestra il modello su un set di dati di addestramento e lo valida su un set di dati di validazione.

    :param model: Modello da addestrare.
    :param criterion: Funzione di perdita.
    :param optimizer: Ottimizzatore.
    :param train_loader: DataLoader per il set di addestramento.
    :param val_loader: DataLoader per il set di validazione.
    :param device: Dispositivo su cui eseguire il training (CPU o GPU).
    :param scheduler: Scheduler per il learning rate.
    :param num_epochs: Numero di epoche per l'addestramento.
    :param model_name: Nome del modello per il salvataggio.
    :return: Migliore accuratezza di validazione e pesi del modello.
    """
    best_val_acc = 0.0
    best_model_weights = None

    for epoch in range(num_epochs):
        start_time = time.time()
        logging.info(f"Inizio dell'epoca {epoch + 1}/{num_epochs}.")

        # Fase di addestramento
        model.train()
        running_loss = 0.0
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient Clipping con max_norm inferiore
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm_value)

            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 10 == 0:
                logging.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoca {epoch + 1} completata. Perdita di addestramento: {epoch_loss:.4f}')

        # Fase di validazione
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
        logging.info(f'Epoca {epoch + 1} - Perdita di validazione: {val_loss:.4f}, Accuratezza di validazione: {val_acc:.4f}')

        end_time = time.time()
        epoch_time = end_time - start_time
        logging.info(f'Tempo di addestramento dell\'epoca {epoch + 1}: {epoch_time:.2f} secondi')

        scheduler.step(val_loss)  # Aggiorna lo scheduler

        # Salva il modello se l'accuratezza di validazione migliora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_weights = model.state_dict()
            torch.save(best_model_weights, f'best_model_{model_name}.pth')
            logging.info(f"L'accuratezza di validazione è migliorata a {val_acc:.4f}. Salvataggio dei pesi del modello...")

    logging.info(f"Addestramento completato. Migliore accuratezza di validazione: {best_val_acc:.4f}")
    return best_val_acc, best_model_weights


@hydra.main(config_path="./configs", config_name="config_transfer_learning")
def main(cfg: DictConfig):
    """
    Funzione principale per l'addestramento del modello con trasferimento di apprendimento.

    :param cfg: Configurazione letta dal file di configurazione.
    """
    logging.info("Inizio del processo di addestramento con trasferimento di apprendimento.")

    # Inizializzazione di CodeCarbon per il monitoraggio delle emissioni di CO2
    tracker = EmissionsTracker()
    tracker.start()

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    val_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.val_path)

    # Caricamento del dataset
    logging.info(f"Caricamento del dataset da {train_path} e {val_path}.")
    train_dataset = ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = ImageFolder(root=val_path, transform=data_transforms['val'])

    # Funzione per creare DataLoader
    def create_data_loader(batch_size):
        logging.info(f"Creazione del DataLoader con batch size: {batch_size}.")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

    # Impostazione degli iperparametri
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 25

    logging.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}, Numero di epoche: {num_epochs}")

    train_loader, val_loader = create_data_loader(batch_size)

    # Carica il modello con pesi pre-addestrati
    if cfg.model.name == 'resnet50':
        logging.info("Caricamento del modello ResNet50 con pesi pre-addestrati.")
        model = models.resnet50(pretrained=True)
    elif cfg.model.name == 'inception_v3':
        logging.info("Caricamento del modello Inception V3 con pesi pre-addestrati.")
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif cfg.model.name == 'xception':
        logging.info("Caricamento del modello Xception con pesi pre-addestrati.")
        model = load_custom_xception_model(pretrained=True)

    model = modify_model(model, cfg.model.name)  # Modifica per immagini in scala di grigi

    # Imposta il dispositivo (CPU o GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Definisci la funzione di perdita e l'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  # Aggiunto weight decay

    # Scheduler per il learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Addestra il modello
    val_acc, _ = train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler,
                             num_epochs=num_epochs)
    logging.info(f"Accuratezza finale di validazione: {val_acc:.4f}")

    # Interrompi il tracker e salva le metriche di emissione
    emissions = tracker.stop()
    logging.info(f"Emissioni di CO2 generate durante l'addestramento: {emissions} kg")


if __name__ == "__main__":
    main()
