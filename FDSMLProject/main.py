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

import codecarbon

param_grid = {
    'learning_rate': [0.01, 0.001, 0.0001],
    'batch_size': [32, 64, 128]
}

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


def load_and_split_data(cfg):
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in listdir(train_path)]))}
    return train_labels, test_path,train_path


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device, num_classes: int):
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
    return torch.topk(output, k=1, dim=-1).indices.flatten()


def test(model: nn.Module, test_loader: DataLoader, loss_function: nn.Module, device):
    model.eval()
    test_loss, correct = 0, 0
    all_predictions, all_gold = [], []

    with torch.no_grad():
        for data, target in test_loader:
            target = target.to(device)
            output = model(data.to(device))
            pred = softmax2predictions(output)

            test_loss += loss_function(output, target).sum().item()
            correct += pred.eq(target.view_as(pred)).sum().item()

            all_predictions.append(pred.cpu().numpy())
            all_gold.append(target.cpu().numpy())

    test_loss /= len(test_loader.dataset)

    logging.info('    Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)

    logging.info("    Acc.: %2.2f%%; F-macro: %2.2f%%\n" % (accuracy_score(y_true, y_pred) * 100,
                                                            f1_score(y_true, y_pred, average="macro") * 100))
    return accuracy_score(y_true, y_pred)


def train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device, epochs,
                       model_name, early_stopping_patience=10):
    best_val_acc = 0.0
    early_stopping_counter = 0

    for epoch in range(epochs):
        start_time = time.time()
        train(model, train_loader, optimizer, loss_function, epoch, device)
        end_time = time.time()

        epoch_time = end_time - start_time
        logging.info(f"Epoch {epoch + 1} training time: {epoch_time:.2f} seconds")

        logging.info("Evaluation on development set:")
        val_acc = test(model, val_loader, loss_function, device)
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

    return best_val_acc


def perform_grid_search(train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, model_class):
    best_params = None
    best_acc = 0.0
    best_optimizer_state = None
    grid_search_results = []

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

            val_acc = train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device,
                                         cfg.model.epochs, model_name=model_class.__name__)

            grid_search_results.append((lr, batch_size, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_optimizer_state = optimizer.state_dict()
                torch.save(model.state_dict(), f"{model_class.__name__}_best_model_weights.pth")

    logging.info(f"Best parameters found: {best_params} with accuracy {best_acc:.2f}%")
    return best_params, grid_search_results, best_optimizer_state


def plot_grid_search_results(grid_search_results, param_grid):
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


@hydra.main(config_path="./configs", config_name="config")
def main_Glyphnet(cfg):
    # Avvio del tracker di CodeCarbon
    tracker = EmissionsTracker(output_dir=cfg.hydra.run.dir)  # Puoi specificare una directory di output se necessario
    tracker.start()

    train_labels, test_path, train_path = load_and_split_data(cfg)

    train_set = GlyphData(root=train_path, class_to_idx=train_labels, transform=train_transform)

    logging.info("Splitting data...")

    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)),
        train_set.targets,
        test_size=cfg.data.val_fraction,
        shuffle=True,
        random_state=cfg.model.seed
    )

    logging.info(f"CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Passare la classe del modello (Glyphnet) come argomento
    best_params, grid_search_results, best_optimizer_state = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, Glyphnet)

    # Dopo la Grid Search: carica il miglior modello e valuta
    test_labels_set = {l for l in [p.strip("/") for p in listdir(test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(root=test_path, class_to_idx=test_labels,
                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                       transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    # Carica il miglior modello trovato durante la Grid Search
    model = Glyphnet(num_classes=len(train_labels))
    model.load_state_dict(torch.load("Glyphnet_best_model_weights.pth"))
    model.to(device)

    logging.info("Checking quality on test set:")
    evaluate_model(model, test_loader, device, num_classes=len(train_labels))

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), "checkpoint.bin")

    logging.info("Saving the optimizer state.")
    torch.save(best_optimizer_state, "optimizer_state.pth")

    plot_grid_search_results(grid_search_results, param_grid)
    tracker.stop()


@hydra.main(config_path="./configs", config_name="config_atcnet")
def main_ATCNet(cfg):
    # Avvio del tracker di CodeCarbon
    tracker = EmissionsTracker(output_dir=cfg.hydra.run.dir)  # Puoi specificare una directory di output se necessario
    tracker.start()

    train_labels, test_path, train_path = load_and_split_data(cfg)

    train_set = datasets.ImageFolder(root=train_path, transform=train_transform)

    logging.info("Splitting data...")

    train_indices, val_indices = train_test_split(range(len(train_set)), test_size=cfg.data.val_fraction,
                                                  random_state=cfg.model.seed)
    logging.info(f"CUDA available? {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Passare la classe del modello (ATCNet) come argomento
    best_params, grid_search_results, best_optimizer_state = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, ATCNet)

    test_set = datasets.ImageFolder(root=test_path, transform=train_transform)
    test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    model = ATCNet(n_classes=len(train_set.classes))

    # Caricamento pesi specifici per ATCNet
    model.load_state_dict(torch.load("ATCNet_best_model_weights.pth"))
    model.to(device)

    logging.info("Checking quality on test set:")
    evaluate_model(model, test_loader, device, num_classes=len(train_set.classes))

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), "ATCNet_checkpoint.bin")

    plot_grid_search_results(grid_search_results, param_grid)

    tracker.stop()


if __name__ == "__main__":

    main_Glyphnet()
