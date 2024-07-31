import logging
import os
from os import listdir
import time
import hydra
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import Subset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import GlyphData  # Import personalizzato del dataset
from model import Glyphnet  # Import del modello personalizzato


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

        # Log ogni batch_reports_interval batch
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
    model.eval()  # Imposta il modello in modalità valutazione
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


def train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device, epochs):
    best_val_acc = 0.0
    for epoch in range(epochs):
        start_time = time.time()
        train(model, train_loader, optimizer, loss_function, epoch, device)
        end_time = time.time()

        epoch_time = end_time - start_time
        logging.info(f"Epoch {epoch + 1} training time: {epoch_time:.2f} seconds")

        logging.info("Evaluation on development set:")
        val_acc = test(model, val_loader, loss_function, device)
        scheduler.step()

        # Salva il modello se la validazione migliora
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logging.info("Validation accuracy improved, saving model...")
            torch.save(model.state_dict(), "best_model_weights.pth")

    return best_val_acc


def perform_grid_search(train_set, train_indices, val_indices, param_grid, train_labels, device, cfg):
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

            model = Glyphnet(num_classes=len(train_labels))
            model.to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, amsgrad=True)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
            loss_function = nn.CrossEntropyLoss()

            val_acc = train_and_evaluate(train_loader, val_loader, model, optimizer, scheduler, loss_function, device,
                                         cfg.model.epochs)

            grid_search_results.append((lr, batch_size, val_acc))

            if val_acc > best_acc:
                best_acc = val_acc
                best_params = {'learning_rate': lr, 'batch_size': batch_size}
                best_optimizer_state = optimizer.state_dict()
                torch.save(model.state_dict(), "best_model_weights.pth")

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
def main(cfg):
    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    train_labels = {l: i for i, l in enumerate(sorted([p.strip("/") for p in listdir(train_path)]))}

    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

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

    param_grid = {
        'learning_rate': [0.01, 0.001, 0.0001],
        'batch_size': [32, 64, 128]
    }

    best_params, grid_search_results, best_optimizer_state = perform_grid_search(train_set, train_indices, val_indices,
                                                                                 param_grid, train_labels, device, cfg)

    test_labels_set = {l for l in [p.strip("/") for p in listdir(test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(root=test_path, class_to_idx=test_labels,
                         transform=transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                                       transforms.ToTensor()]))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)
    model = Glyphnet(num_classes=len(train_labels))
    model.load_state_dict(torch.load("best_model_weights.pth"))
    model.to(device)
    model.eval()

    logging.info("Checking quality on test set:")
    test(model, test_loader, nn.CrossEntropyLoss(), device)

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), "checkpoint.bin")

    logging.info("Saving the optimizer state.")
    torch.save(best_optimizer_state, "optimizer_state.pth")

    plot_grid_search_results(grid_search_results, param_grid)


if __name__ == "__main__":
    main()
