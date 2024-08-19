import logging
import os
import numpy as np
import torch
from codecarbon import EmissionsTracker
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import hydra
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig
import torch.nn.functional as F
import timm

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definizione delle trasformazioni per il dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizzazione per immagini RGB
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizzazione per immagini RGB
    ]),
}


def train_model(model, criterion, optimizer, train_loader, val_loader, device, scheduler, num_epochs=25,
                early_stopping_patience=10):
    model.to(device)
    best_val_acc = 0.0
    patience_counter = 0
    best_model_weights = None  # Variabile per salvare i pesi migliori

    logging.info("Inizio training del modello...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        logging.info(f'Epoch {epoch + 1}/{num_epochs}')

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

            if batch_idx % 10 == 0:  # Log ogni 10 batch
                logging.info(f'Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

        epoch_loss = running_loss / len(train_loader.dataset)
        logging.info(f'Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}')

        model.eval()
        val_loss = 0.0
        corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data).cpu().numpy()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects / len(val_loader.dataset)
        logging.info(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}')

        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_model_weights = model.state_dict()  # Salva i pesi migliori
            logging.info("Validation accuracy improved, saving model weights...")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info("Early stopping triggered")
                break

    # Alla fine dell'allenamento, ritorna i migliori pesi trovati
    model.load_state_dict(best_model_weights)
    return best_val_acc, best_model_weights


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss, correct = 0, 0
    all_predictions, all_gold = [], []

    logging.info("Inizio test del modello...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).cpu().numpy()
            all_predictions.append(preds.cpu().numpy())
            all_gold.append(labels.cpu().numpy())

            if batch_idx % 10 == 0:  # Log ogni 10 batch
                logging.info(f'Test Batch {batch_idx}/{len(test_loader)} - Loss: {loss.item():.4f}')

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)
    logging.info(f'Average test loss: {test_loss:.4f}, Accuracy: {test_acc * 100:.2f}%')

    y_pred = np.concatenate(all_predictions)
    y_true = np.concatenate(all_gold)

    return accuracy_score(y_true, y_pred)


def calculate_model_weights(models, val_loader, device):
    logging.info("Calcolo dei pesi per i modelli in base alle performance...")
    weights = []
    for model in models:
        val_acc = test(model, val_loader, nn.CrossEntropyLoss(), device)
        weights.append(val_acc)
    return np.array(weights) / np.sum(weights)


def predict_ensemble(models, data_loader, device, weights=None):
    logging.info("Combinazione delle predizioni tramite soft voting...")
    all_probabilities = []
    for model in models:
        model.to(device)
        model.eval()
        model_probabilities = []
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = F.softmax(outputs, dim=1).cpu().numpy()
                model_probabilities.append(probabilities)
        all_probabilities.append(np.concatenate(model_probabilities))

    # Combina le probabilità (soft voting)
    if weights is not None:
        all_probabilities = np.average(all_probabilities, axis=0, weights=weights)
    else:
        all_probabilities = np.mean(all_probabilities, axis=0)

    combined_predictions = np.argmax(all_probabilities, axis=1)
    return combined_predictions


@hydra.main(config_path="./configs", config_name="config_ensamble_learning")
def main(cfg: DictConfig):
    logging.info("Inizio pipeline di addestramento...")

    tracker = EmissionsTracker()  # Inizializza il tracker di CodeCarbon
    tracker.start()  # Avvia il monitoraggio delle emissioni

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    val_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.val_path)

    logging.info(f'Caricamento del dataset da {train_path} e {val_path}...')

    # Creare directory per salvare i file
    save_dir = os.path.join(hydra.utils.get_original_cwd(), 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    logging.info(f"Files will be saved in: {save_dir}")

    # Caricamento del dataset
    train_dataset = datasets.ImageFolder(root=train_path, transform=data_transforms['train'])
    val_dataset = datasets.ImageFolder(root=val_path, transform=data_transforms['val'])

    # Creazione dei DataLoader per training e validation
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(train_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)

    tresnet_models = []
    model_weights = []
    all_model_weights = []  # Lista per salvare i pesi di tutti i modelli

    # Addestramento di modelli TResNet
    logging.info("Inizio addestramento dei modelli TResNet...")

    for fold in range(2):  # Ad esempio, 4 modelli
        logging.info(f"Training TResNet model {fold + 1}/4")

        model = timm.create_model('tresnet_m', pretrained=True, num_classes=len(train_dataset.classes)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

        best_val_acc, best_weights = train_model(model, criterion, optimizer, train_loader, val_loader, device,
                                                 scheduler,
                                                 num_epochs=cfg.train.num_epochs, early_stopping_patience=10)
        tresnet_models.append(model)
        model_weights.append(best_val_acc)
        all_model_weights.append(best_weights)  # Aggiungi i pesi migliori alla lista

    # Calcola i pesi per il weighted voting
    logging.info("Calcolo dei pesi per il weighted voting...")
    model_weights = calculate_model_weights(tresnet_models, val_loader, device)

    # Combina le predizioni usando il weighted soft voting
    logging.info("Inizio delle predizioni combinate con weighted soft voting...")
    combined_predictions = predict_ensemble(tresnet_models, val_loader, device, weights=model_weights)

    true_labels = np.array([label for _, label in val_loader.dataset])

    # Calcola le metriche di valutazione
    logging.info("Calcolo delle metriche di valutazione finali...")
    final_accuracy = accuracy_score(true_labels, combined_predictions) * 100
    precision = precision_score(true_labels, combined_predictions, average='weighted') * 100
    recall = recall_score(true_labels, combined_predictions, average='weighted') * 100
    f1 = f1_score(true_labels, combined_predictions, average='weighted') * 100

    logging.info(f'Final ensemble accuracy: {final_accuracy:.2f}%')
    logging.info(f'Precision: {precision:.2f}%')
    logging.info(f'Recall: {recall:.2f}%')
    logging.info(f'F1-Score: {f1:.2f}%')

    # Confusion Matrix
    logging.info("Plot della confusion matrix...")
    cm = confusion_matrix(true_labels, combined_predictions)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix of Ensemble Model')
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'))  # Salva la confusion matrix
    plt.show()

    # Salva i pesi finali di ogni modello
    for i, weights in enumerate(all_model_weights):
        model_path = os.path.join(save_dir, f'model_weights_fold_{i + 1}.pt')
        torch.save(weights, model_path)
        logging.info(f'Saved model weights for fold {i + 1} at {model_path}')

    # Salva i pesi dell'ensemble combinato (opzionale)
    ensemble_path = os.path.join(save_dir, 'ensemble_model.pth')
    torch.save({'models': [model.state_dict() for model in tresnet_models],
                'weights': model_weights}, ensemble_path)
    logging.info(f'Saved ensemble model at {ensemble_path}')

    emissions = tracker.stop()  # Ferma il monitoraggio e ottieni le emissioni
    logging.info(f"Emissioni di CO2 generate durante l'addestramento: {emissions} kg")
    emissions_file = os.path.join(save_dir, 'emissions.txt')
    with open(emissions_file, 'w') as f:
        f.write(f"Emissioni di CO2: {emissions} kg")
    logging.info(f"Emissions saved at {emissions_file}")


if __name__ == "__main__":
    main()
