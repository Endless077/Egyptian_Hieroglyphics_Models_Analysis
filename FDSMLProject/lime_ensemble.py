import os
import torch
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import timm
import torch.nn.functional as F
import hydra
from omegaconf import DictConfig
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Definizione delle trasformazioni per il dataset
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalizzazione per immagini RGB
])


def load_ensemble_model(ensemble_model_path, device, num_classes):
    checkpoint = torch.load(ensemble_model_path)
    models = []

    for model_state_dict in checkpoint['models']:
        model = timm.create_model('tresnet_m', pretrained=True, num_classes=num_classes)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()
        models.append(model)

    model_weights = checkpoint.get('weights', None)
    return models, model_weights


def predict_ensemble_lime(models, image, device, weights=None):
    all_probabilities = []

    # Assicurati che l'immagine abbia 4 dimensioni
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    for model in models:
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()
        all_probabilities.append(probabilities)

    if weights is not None:
        all_probabilities = np.average(all_probabilities, axis=0, weights=weights)
    else:
        all_probabilities = np.mean(all_probabilities, axis=0)

    return all_probabilities


def lime_explanation(image_path, models, model_weights, device, true_class=None):
    # Caricamento dell'immagine da interpretare
    logging.info(f"Caricamento dell'immagine da interpretare: {image_path}")
    image = Image.open(image_path)
    image = data_transforms(image)

    # Funzione predizione per LIME
    def predict_fn(images):
        # Converti l'immagine in un formato compatibile con PIL
        images = [np.uint8(255 * img) if img.dtype == np.float32 else img for img in images]

        # Applica le trasformazioni e impila le immagini
        images = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0)

        # Assicura che l'input abbia la dimensione corretta
        if len(images.shape) == 3:
            images = images.unsqueeze(0)  # Aggiungi la dimensione del batch se manca

        predictions = predict_ensemble_lime(models, images, device, weights=model_weights)
        return predictions

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image.permute(1, 2, 0)),
                                             predict_fn,
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    # Visualizza la classe prevista e quella corretta
    predicted_class = explanation.top_labels[0]
    logging.info(f"Classe predetta: {predicted_class}")
    if true_class is not None:
        logging.info(f"Classe corretta: {true_class}")

    # Recupera l'oggetto ImageExplanation e usa get_image_and_mask
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10,
                                                hide_rest=False)

    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(temp, mask))
    plt.title(f'LIME Explanation for class {predicted_class}')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return predicted_class


@hydra.main(config_path="./configs", config_name="config_lime")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Caricamento del modello ensemble da {cfg.ensemble_model_path}...")
    models, model_weights = load_ensemble_model(cfg.ensemble_model_path, device, cfg.num_classes)

    # Carica l'immagine e ottieni la predizione
    predicted_class = lime_explanation(cfg.image_path, models, model_weights, device, true_class=cfg.true_class)


if __name__ == "__main__":
    main()
