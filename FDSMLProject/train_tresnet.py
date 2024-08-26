import hydra
from codecarbon import EmissionsTracker
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import os
import time
import copy
import logging
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configurazione del logger
log = logging.getLogger(__name__)


# Funzione per caricare il dataset
def setup_data_loaders(train_path, valid_path, batch_size, img_size):
    """
    Configura i DataLoader per il training e la validazione.

    Args:
        train_path (str): Percorso del dataset di training.
        valid_path (str): Percorso del dataset di validazione.
        batch_size (int): Dimensione del batch.
        img_size (int): Dimensione dell'immagine.

    Returns:
        dict: Dataloaders per il training e la validazione.
        dict: Dimensioni dei dataset.
        list: Nomi delle classi.
    """
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

    image_datasets = {
        'train': datasets.ImageFolder(train_path, data_transforms['train']),
        'valid': datasets.ImageFolder(valid_path, data_transforms['valid'])
    }

    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'valid': DataLoader(image_datasets['valid'], batch_size=batch_size, shuffle=True, num_workers=4)
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    return dataloaders, dataset_sizes, image_datasets['train'].classes


# Funzione di addestramento del modello
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    """
    Addestra il modello e ne salva i pesi migliori.

    Args:
        model (torch.nn.Module): Modello da addestrare.
        criterion (torch.nn.Module): Funzione di perdita.
        optimizer (torch.optim.Optimizer): Ottimizzatore.
        scheduler (torch.optim.lr_scheduler): Scheduler per il learning rate.
        dataloaders (dict): Dataloaders per il training e la validazione.
        dataset_sizes (dict): Dimensioni dei dataset.
        device (torch.device): Dispositivo su cui addestrare il modello.
        num_epochs (int): Numero di epoche di addestramento.

    Returns:
        model: Modello con i migliori pesi.
        list: Perdite durante il training.
        list: Perdite durante la validazione.
        list: Accuratezze durante il training.
        list: Accuratezze durante la validazione.
    """
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []

    for epoch in range(num_epochs):
        log.info(f'Epoch {epoch}/{num_epochs - 1}')
        log.info('-' * 10)

        # Ogni epoca ha una fase di training e una di validazione
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Imposta il modello in modalità training
            else:
                model.eval()  # Imposta il modello in modalità valutazione

            running_loss = 0.0
            running_corrects = 0

            # Loop sui dati
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            log.info(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Salva le perdite e le accuratezze
            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc.item())
            else:
                valid_losses.append(epoch_loss)
                valid_accuracies.append(epoch_acc.item())

            # Salva i pesi migliori del modello
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        log.info('')

    time_elapsed = time.time() - since
    log.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    log.info(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model, train_losses, valid_losses, train_accuracies, valid_accuracies


# Funzione per il training con parametri specifici
def train_with_params(cfg, learning_rate, batch_size):
    """
    Configura e addestra il modello con parametri specifici.

    Args:
        cfg (DictConfig): Configurazione Hydra.
        learning_rate (float): Learning rate da utilizzare.
        batch_size (int): Dimensione del batch.

    Returns:
        model: Modello addestrato.
        optimizer: Ottimizzatore configurato.
        scheduler: Scheduler configurato.
        dataloaders: Dataloaders per il training e la validazione.
        dataset_sizes: Dimensioni dei dataset.
        class_names: Nomi delle classi.
        train_losses: Perdite durante il training.
        valid_losses: Perdite durante la validazione.
        train_accuracies: Accuratezze durante il training.
        valid_accuracies: Accuratezze durante la validazione.
    """
    # Configurazione del dispositivo
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    valid_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.valid_path)

    # Configurazione data loaders con batch_size specifico
    dataloaders, dataset_sizes, class_names = setup_data_loaders(train_path, valid_path, batch_size, cfg.train.img_size)

    # Caricamento del modello
    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained, num_classes=len(class_names))
    model = model.to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler opzionale per ridurre il learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler_step_size,
                                          gamma=cfg.train.scheduler_gamma)

    # Avvio dell'addestramento
    model, train_losses, valid_losses, train_accuracies, valid_accuracies = train_model(
        model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, cfg.train.num_epochs)

    return model, optimizer, scheduler, dataloaders, dataset_sizes, class_names, train_losses, valid_losses, train_accuracies, valid_accuracies


# Funzione per valutare il modello e calcolare le metriche
def evaluate_model(model, dataloaders, dataset_sizes, device):
    """
    Valuta il modello e calcola le metriche di precisione, recall e f1.

    Args:
        model (torch.nn.Module): Modello da valutare.
        dataloaders (dict): Dataloaders per la validazione.
        dataset_sizes (dict): Dimensioni dei dataset.
        device (torch.device): Dispositivo su cui eseguire la valutazione.

    Returns:
        float: Accuratezza del modello.
        float: Precisione del modello.
        float: Recall del modello.
        float: F1 score del modello.
        numpy.ndarray: Matrice di confusione.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['valid']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = np.mean(all_preds == all_labels)

    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, precision, recall, f1, cm


# Funzione per salvare la matrice di confusione
def save_confusion_matrix(cm, class_names, output_dir):
    """
    Salva la matrice di confusione come immagine.

    Args:
        cm (numpy.ndarray): Matrice di confusione.
        class_names (list): Nomi delle classi.
        output_dir (str): Directory di output dove salvare l'immagine.

    Returns:
        None
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()


# Funzione per salvare le curve di apprendimento
def save_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies, output_dir):
    """
    Salva le curve di apprendimento (perdite e accuratezze) come immagine.

    Args:
        train_losses (list): Perdite durante il training.
        valid_losses (list): Perdite durante la validazione.
        train_accuracies (list): Accuratezze durante il training.
        valid_accuracies (list): Accuratezze durante la validazione.
        output_dir (str): Directory di output dove salvare l'immagine.

    Returns:
        None
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, valid_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy')
    plt.plot(epochs, valid_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curves')
    plt.legend()

    plt.savefig(os.path.join(output_dir, 'learning_curves.png'))
    plt.close()


@hydra.main(config_path="./configs", config_name="config_tresnet")
def main(cfg: DictConfig):
    """
    Funzione principale per eseguire il training e la valutazione di un modello utilizzando grid search.

    Args:
        cfg (DictConfig): Configurazione Hydra.

    Returns:
        None
    """
    # Avvio del tracker di CodeCarbon per monitorare le emissioni
    tracker = EmissionsTracker()
    tracker.start()

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    best_model_wts = None
    best_acc = 0.0
    best_params = {}
    best_metrics = {}

    # Esegui una grid search sui learning rate e batch size specificati
    for lr in cfg.grid_search.learning_rates:
        for batch_size in cfg.grid_search.batch_sizes:
            log.info(f"Training with learning rate: {lr} and batch size: {batch_size}")

            # Configura il modello per questa combinazione di parametri
            model, optimizer, scheduler, dataloaders, dataset_sizes, class_names, train_losses, valid_losses, train_accuracies, valid_accuracies = train_with_params(
                cfg, lr, batch_size)

            # Calcolare e valutare le metriche
            accuracy, precision, recall, f1, cm = evaluate_model(model, dataloaders, dataset_sizes, device)
            log.info(f"Validation Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

            # Salva i pesi migliori se questo modello è il migliore finora
            if accuracy > best_acc:
                best_acc = accuracy
                best_model_wts = copy.deepcopy(model.state_dict())
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_metrics = {'precision': precision, 'recall': recall, 'f1': f1, 'confusion_matrix': cm}
                best_train_losses = train_losses
                best_valid_losses = valid_losses
                best_train_accuracies = train_accuracies
                best_valid_accuracies = valid_accuracies

    # Dopo la grid search, carica il modello con i migliori pesi trovati
    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained, num_classes=len(class_names))
    model = model.to(device)
    model.load_state_dict(best_model_wts)

    # Salvataggio del miglior modello
    output_dir = Path(hydra.utils.get_original_cwd()) / 'saved_models'
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / cfg.checkpoint_name
    torch.save(model.state_dict(), model_path)
    log.info(f"Best model saved to {model_path} with parameters: {best_params} and validation accuracy: {best_acc:.4f}")

    # Salvataggio della matrice di confusione
    save_confusion_matrix(best_metrics['confusion_matrix'], class_names, output_dir)

    # Salvataggio delle curve di apprendimento
    save_learning_curves(best_train_losses, best_valid_losses, best_train_accuracies, best_valid_accuracies, output_dir)

    # Ferma il tracker e registra le emissioni
    tracker.stop()


if __name__ == '__main__':
    main()
