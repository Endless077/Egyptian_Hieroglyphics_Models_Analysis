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

# Configurazione del logger
log = logging.getLogger(__name__)


# Funzione per caricare il dataset
def setup_data_loaders(train_path, valid_path, batch_size, img_size):
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


# Funzione di addestramento
def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        log.info(f'Epoch {epoch}/{num_epochs - 1}')
        log.info('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

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

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        log.info('')

    time_elapsed = time.time() - since
    log.info(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    log.info(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


@hydra.main(config_path="./configs", config_name="config_tresnet")
def main(cfg: DictConfig):
    # Avvio del tracker di CodeCarbon
    tracker = EmissionsTracker(output_dir=cfg.hydra.run.dir)  # Puoi specificare una directory di output se necessario
    tracker.start()

    # Configurazione del dispositivo
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.train_path)
    valid_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.valid_path)
    test_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.test_path)

    # Configurazione data loaders
    dataloaders, dataset_sizes, class_names = setup_data_loaders(train_path, valid_path,
                                                                 cfg.train.batch_size, cfg.train.img_size)

    # Caricamento del modello
    model = timm.create_model(cfg.model.model_name, pretrained=cfg.model.pretrained, num_classes=len(class_names))
    model = model.to(device)

    # Definizione della funzione di perdita e dell'ottimizzatore
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.learning_rate)

    # Scheduler opzionale per ridurre il learning rate
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.scheduler_step_size,
                                          gamma=cfg.train.scheduler_gamma)

    # Avvio dell'addestramento
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device,
                        cfg.train.num_epochs)

    # Creazione della directory di output
    output_dir = Path(cfg.hydra.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Salvataggio del modello
    model_path = output_dir / cfg.checkpoint_name
    torch.save(model.state_dict(), model_path)
    log.info(f"Model saved to {model_path}")

    # Ferma il tracker e registra le emissioni
    tracker.stop()


if __name__ == '__main__':
    main()
