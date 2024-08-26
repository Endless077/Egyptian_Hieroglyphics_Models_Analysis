import logging
import os
from os import listdir
import time
import hydra
import numpy as np
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import (f1_score, accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, log_loss)
import seaborn as sns
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

from ATCNet import ATCNet
from dataset import GlyphData
from model import Glyphnet

# Definisce una griglia di iperparametri per la ricerca
param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128]
}

# Trasformazioni per l'addestramento dei dati
train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def load_and_split_data(cfg):
    """
    Carica i dati di addestramento e test e crea una mappatura delle etichette per i dati di addestramento.

    Args:
        cfg (DictConfig): Configurazione di Hydra contenente i percorsi dei dati.

    Returns:
        Tuple[Dict[str, int], str, str]: Mappatura delle etichette di addestramento, percorso dei dati di test, percorso dei dati di addestramento.
    """
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    # Crea una mappatura delle etichette di addestramento
    train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in listdir(train_path)]))}
    return train_labels, test_path, train_path


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, num_classes: int):
    """
    Valuta il modello su un set di test e calcola le metriche di valutazione.

    Args:
        model (nn.Module): Il modello da valutare.
        test_loader (DataLoader): DataLoader contenente i dati di test.
        device (torch.device): Il dispositivo su cui eseguire il modello (CPU o GPU).
        num_classes (int): Numero di classi nel dataset.

    Returns:
        None
    """
    model.eval()
    all_predictions, all_gold = [], []
    all_logits = []

    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output = model(data.to(device))
            pred = softmax2predictions(output)

            all_predictions.append(pred.cpu().numpy())
            all_gold.append(target.cpu().numpy())
            all_logits.append(F.softmax(output, dim=1).cpu().numpy())

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)
    y_logits = np.concatenate(all_logits)

    # Calcolo delle metriche
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

    # Log delle metriche
    logging.info(
        f"    Acc.: {acc * 100:.2f}%; Precision: {precision * 100:.2f}%; Recall: {recall * 100:.2f}%; F1: {f1 * 100:.2f}%")
    if roc_auc is not None:
        logging.info(f"    ROC-AUC: {roc_auc:.4f}")
    logging.info(f"    Log Loss: {logloss:.4f}")

    # Plot della confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def train(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer,
          loss_function: nn.Module, current_epoch_number: int = 0,
          device: torch.device = None, batch_reports_interval: int = 100):
    """
    Addestra il modello su un epoch.

    Args:
        model (nn.Module): Il modello da addestrare.
        train_loader (DataLoader): DataLoader contenente i dati di addestramento.
        optimizer (optim.Optimizer): Ottimizzatore per aggiornare i pesi del modello.
        loss_function (nn.Module): Funzione di perdita da utilizzare.
        current_epoch_number (int): Numero dell'epoch corrente.
        device (torch.device): Il dispositivo su cui eseguire il modello (CPU o GPU).
        batch_reports_interval (int): Numero di batch dopo i quali loggare lo stato.

    Returns:
        None
    """
    model.train()  # Imposta il modello in modalità addestramento
    loss_accum = 0  # Accumulatore per la perdita

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # Azzeramento dei gradienti
        output = model(data.to(device))  # Predizione dei punteggi
        loss = loss_function(output, target.to(device))  # Calcolo dell'errore
        loss_accum += loss.item() / len(data)  # Salvataggio della perdita per le statistiche
        loss.backward()  # Calcolo dei gradienti
        optimizer.step()  # Aggiornamento dei pesi del modello

        if batch_idx % batch_reports_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tAveraged Epoch Loss: {:.6f}'.format(
                current_epoch_number + 1,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss_accum / (batch_idx + 1)))


def softmax2predictions(output: torch.Tensor) -> torch.Tensor:
    """
    Converte l'output softmax del modello in predizioni di classe.

    Args:
        output (torch.Tensor): Output del modello.

    Returns:
        torch.Tensor: Indici delle classi predette.
    """
    return torch.topk(output, k=1, dim=-1).indices.flatten()


def validate(model: nn.Module, val_loader: DataLoader, loss_function: nn.Module, device):
    """
    Valida il modello sul set di validazione e calcola le metriche.

    Args:
        model (nn.Module): Il modello da validare.
        val_loader (DataLoader): DataLoader contenente i dati di validazione.
        loss_function (nn.Module): Funzione di perdita da utilizzare.
        device (torch.device): Il dispositivo su cui eseguire il modello (CPU o GPU).

    Returns:
        Tuple[float, np.ndarray, np.ndarray]: Perdita media su tutto il set di validazione, etichette vere, predizioni.
    """
    model.eval()
    val_loss = 0.0
    all_predictions, all_gold = [], []

    with torch.no_grad():
        for data, target in val_loader:
            target = target.to(device)
            output = model(data.to(device))
            pred = softmax2predictions(output)

            val_loss += loss_function(output, target).sum().item()
            all_predictions.append(pred.cpu().numpy())
            all_gold.append(target.cpu().numpy())

    val_loss /= len(val_loader.dataset)

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)

    acc = accuracy_score(y_true, y_pred)
    logging.info('    Validation loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        val_loss, np.sum(y_pred == y_true), len(val_loader.dataset),
        100. * acc))

    return val_loss, y_true, y_pred


def train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device, epochs,
                       model_name, early_stopping_patience=10):
    """
    Addestra e valuta il modello utilizzando un meccanismo di early stopping.

    Args:
        train_loader (DataLoader): DataLoader contenente i dati di addestramento.
        val_loader (DataLoader): DataLoader contenente i dati di validazione.
        model (nn.Module): Il modello da addestrare e valutare.
        optimizer (optim.Optimizer): Ottimizzatore per aggiornare i pesi del modello.
        scheduler (optim.lr_scheduler._LRScheduler): Scheduler per aggiornare il tasso di apprendimento.
        loss_function (nn.Module): Funzione di perdita da utilizzare.
        device (torch.device): Il dispositivo su cui eseguire il modello (CPU o GPU).
        epochs (int): Numero di epoche per cui addestrare il modello.
        model_name (str): Nome del modello per salvare i pesi.
        early_stopping_patience (int): Numero di epoche senza miglioramenti dopo cui fermare l'addestramento.

    Returns:
        Dict[str, List[float]]: Metriche raccolte durante l'addestramento e la validazione.
    """
    best_val_acc = 0.0
    early_stopping_counter = 0

    # Liste per salvare le metriche
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start_time = time.time()

        # Fase di addestramento
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data.to(device))
            loss = loss_function(output, target.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            y_pred_train.append(softmax2predictions(output).cpu().numpy())
            y_true_train.append(target.cpu().numpy())

        y_pred_train = np.concatenate(y_pred_train)
        y_true_train = np.concatenate(y_true_train)

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1 = f1_score(y_true_train, y_pred_train, average='macro', zero_division=0)
        train_precision = precision_score(y_true_train, y_pred_train, average='macro', zero_division=0)
        train_recall = recall_score(y_true_train, y_pred_train, average='macro', zero_division=0)

        # Salvare le metriche di addestramento
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_f1_scores.append(train_f1)
        train_precisions.append(train_precision)
        train_recalls.append(train_recall)

        end_time = time.time()
        logging.info(f"Epoch {epoch + 1} training time: {end_time - start_time:.2f} seconds")

        # Fase di validazione
        logging.info("Evaluation on development set:")
        val_loss, y_true_val, y_pred_val = validate(model, val_loader, loss_function, device)

        val_f1 = f1_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_precision = precision_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_recall = recall_score(y_true_val, y_pred_val, average='macro', zero_division=0)
        val_acc = accuracy_score(y_true_val, y_pred_val)

        # Salvare le metriche di validazione
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        val_f1_scores.append(val_f1)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)

        scheduler.step()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stopping_counter = 0
            logging.info(f"Validation accuracy improved, saving {model_name} model...")
            torch.save(model.state_dict(), f"{model_name}_best_model_weights.pth")
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break

    return {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_f1_scores': train_f1_scores,
        'val_f1_scores': val_f1_scores,
        'train_precisions': train_precisions,
        'val_precisions': val_precisions,
        'train_recalls': train_recalls,
        'val_recalls': val_recalls
    }


def plot_learning_curves(m):
    """
    Plot delle curve di apprendimento per accuracy e loss.

    Args:
        m (Dict[str, List[float]]): Metriche raccolte durante l'addestramento e la validazione.

    Returns:
        None
    """
    epochs = range(1, len(m['train_accuracies']) + 1)

    # Plot per la loss
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, m['train_losses'], label='Train Loss')
    plt.plot(epochs, m['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Learning Curve - Loss')
    plt.legend()
    plt.grid(True)

    # Plot per l'accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, m['train_accuracies'], label='Train Accuracy')
    plt.plot(epochs, m['val_accuracies'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve - Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def perform_grid_search(train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, model_class):
    """
    Esegue una ricerca a griglia (grid search) per trovare i migliori iperparametri di addestramento.

    Args:
        train_set (Dataset): Il dataset di addestramento completo.
        train_indices (List[int]): Indici per il set di addestramento.
        val_indices (List[int]): Indici per il set di validazione.
        param_grid (Dict[str, List]): Griglia di iperparametri da testare.
        train_labels (Dict[str, int]): Mappatura delle etichette di addestramento.
        device (torch.device): Il dispositivo su cui eseguire il modello (CPU o GPU).
        cfg (DictConfig): Configurazione di Hydra contenente i parametri di addestramento.
        model_class (nn.Module): Classe del modello da addestrare.

    Returns:
        Tuple[Dict[str, float], List[Tuple[float, float, float]], Dict, Dict]: Migliori iperparametri, risultati della grid search, stato dell'ottimizzatore, metriche migliori.
    """
    best_params = None
    best_acc = 0.0
    best_optimizer_state = None
    grid_search_results = []
    best_metrics = None

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            logging.info(f"Testing combination: lr={lr}, batch_size={batch_size}")

            train_loader = torch.utils.data.DataLoader(Subset(train_set, train_indices),
                                                       batch_size=batch_size,
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(Subset(train_set, val_indices),
                                                     shuffle=False,
                                                     batch_size=batch_size)

            # Gestisci i parametri per il costruttore in base al modello
            if model_class == Glyphnet:
                model = model_class(num_classes=len(train_labels))
            elif model_class == ATCNet:
                model = model_class(n_classes=len(train_labels))
            else:
                raise ValueError(f"Model class {model_class} not recognized")

            model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            loss_function = nn.CrossEntropyLoss()

            metrics = train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device,
                                         cfg.model.epochs, model_name=model_class.__name__)

            if metrics is None:
                logging.warning(f"No metrics were returned for combination: lr={lr}, batch_size={batch_size}")
            else:
                logging.info(f"Metrics for lr={lr}, batch_size={batch_size}: {metrics}")

            val_acc = metrics['val_accuracies'][-1] if metrics is not None else 0
            grid_search_results.append((lr, batch_size, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_optimizer_state = optimizer.state_dict()
                best_metrics = metrics
                torch.save(model.state_dict(), f"{model_class.__name__}_best_model_weights.pth")

    logging.info(f"Best parameters found: {best_params} with accuracy {best_acc:.2f}%")
    return best_params, grid_search_results, best_optimizer_state, best_metrics  # Restituisci anche le metriche


def plot_grid_search_results(grid_search_results, param_grid):
    """
    Plot dei risultati della grid search per visualizzare l'accuratezza in funzione del batch size.

    Args:
        grid_search_results (List[Tuple[float, float, float]]): Risultati della grid search.
        param_grid (Dict[str, List]): Griglia di iperparametri utilizzata.

    Returns:
        None
    """
    lrs, batch_sizes, accuracies = zip(*grid_search_results)
    plt.figure(figsize=(10, 6))
    for lr in param_grid['learning_rate']:
        plt.plot([bs for l, bs, acc in grid_search_results if l == lr],
                 [acc for l, bs, acc in grid_search_results if l == lr],
                 label=f"Learning rate {lr}")

    plt.xlabel('Batch Size')
    plt.ylabel('Validation Accuracy')
    plt.title('Grid Search Results')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

