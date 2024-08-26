import torch
import numpy as np
from lime import lime_image
from skimage.color import gray2rgb
from skimage.segmentation import mark_boundaries, slic
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
    """
    Carica il modello ensemble salvato e i suoi pesi dal percorso specificato.

    Args:
        ensemble_model_path (str): Il percorso del file contenente il modello ensemble.
        device (torch.device): Il dispositivo su cui caricare il modello (CPU o GPU).
        num_classes (int): Numero di classi nel dataset.

    Returns:
        models (list): Lista di modelli TResNet caricati.
        model_weights (list or None): Lista di pesi dei modelli per il soft voting.
    """
    checkpoint = torch.load(ensemble_model_path)
    models = []

    for model_state_dict in checkpoint['models']:
        model = timm.create_model('tresnet_m', pretrained=True, num_classes=num_classes)
        model.load_state_dict(model_state_dict)
        model.to(device)
        model.eval()  # Imposta il modello in modalità valutazione
        models.append(model)

    model_weights = checkpoint.get('weights', None)
    return models, model_weights


def predict_ensemble_lime(models, image, device, weights=None):
    """
    Esegue la predizione su un'immagine utilizzando l'ensemble di modelli, con opzione di soft voting.

    Args:
        models (list): Lista di modelli da utilizzare per la predizione.
        image (torch.Tensor): L'immagine di input.
        device (torch.device): Il dispositivo su cui eseguire la predizione (CPU o GPU).
        weights (list, optional): Pesi per il soft voting tra i modelli. Default è None.

    Returns:
        np.array: Le probabilità combinate delle classi previste dall'ensemble di modelli.
    """
    all_probabilities = []

    # Assicurati che l'immagine abbia 4 dimensioni (batch size incluso)
    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)

    for model in models:
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()  # Calcolo delle probabilità
        all_probabilities.append(probabilities)

    # Combina le probabilità usando i pesi, se forniti
    if weights is not None:
        all_probabilities = np.average(all_probabilities, axis=0, weights=weights)
    else:
        all_probabilities = np.mean(all_probabilities, axis=0)

    return all_probabilities


def perturb_image(image, segments):
    """
    Crea versioni perturbate dell'immagine basate sui segmenti.

    Args:
        image (np.array): L'immagine di input.
        segments (np.array): L'array dei segmenti creati dall'algoritmo SLIC.

    Returns:
        np.array: Array contenente le immagini perturbate.
    """
    perturbed_images = []
    num_segments = np.max(segments) + 1

    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i
        if np.any(mask):
            mean_value = np.mean(image[mask], axis=0)
            std_value = np.std(image[mask], axis=0)
            perturbed_image[mask] = np.random.normal(loc=mean_value, scale=std_value)
        perturbed_images.append(perturbed_image)

    return np.array(perturbed_images)


def custom_explanation(image_path, models, model_weights, device, true_class=None):
    """
    Genera una spiegazione personalizzata utilizzando l'alterazione dei segmenti dell'immagine.

    Args:
        image_path (str): Il percorso dell'immagine da interpretare.
        models (list): Lista di modelli ensemble da usare per la predizione.
        model_weights (list or None): Pesi dei modelli per il soft voting.
        device (torch.device): Il dispositivo su cui eseguire la predizione (CPU o GPU).
        true_class (int, optional): La classe corretta dell'immagine, se conosciuta.

    Returns:
        int: La classe prevista dall'ensemble.
    """
    logging.info(f"Caricamento dell'immagine da interpretare: {image_path}")
    image = Image.open(image_path)
    image = np.array(image)

    # Se l'immagine è in scala di grigi, convertila in RGB
    if len(image.shape) == 2:
        image = gray2rgb(image)

    # Segmenta l'immagine in superpixel
    segments = slic(image, n_segments=150, compactness=10)

    # Prepara l'immagine base per la predizione
    base_image_tensor = data_transforms(Image.fromarray(image)).unsqueeze(0).to(device)

    # Predizione utilizzando l'ensemble di modelli
    base_prediction = predict_ensemble_lime(models, base_image_tensor, device, model_weights)
    top_label = np.argmax(base_prediction[0])  # Estrai la classe con la probabilità maggiore
    logging.info(f"Classe predetta: {top_label}")
    if true_class is not None:
        logging.info(f"Classe corretta: {true_class}")

    # Crea versioni perturbate dell'immagine
    perturbed_images = perturb_image(image, segments)

    # Preprocessa le immagini perturbate e predici
    perturbed_images_tensor = torch.cat(
        [data_transforms(Image.fromarray(img)).unsqueeze(0) for img in perturbed_images], dim=0).to(device)
    predictions = predict_ensemble_lime(models, perturbed_images_tensor, device, model_weights)

    # Calcola i pesi in base alle predizioni per la classe principale
    weights = np.mean(predictions[:, top_label], axis=0)

    # Genera la spiegazione mappando i pesi ai segmenti
    explanation = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        explanation[segments == i] = weights

    # Visualizza la spiegazione sovrapposta all'immagine
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(image / 255.0, segments))
    plt.imshow(explanation, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.title(f"Custom Explanation for class {top_label}")
    plt.axis('off')
    plt.show()

    return top_label


def lime_library_explanation(image_path, models, model_weights, device, true_class=None):
    """
    Genera una spiegazione usando la libreria LIME.

    Args:
        image_path (str): Il percorso dell'immagine da interpretare.
        models (list): Lista di modelli ensemble da usare per la predizione.
        model_weights (list or None): Pesi dei modelli per il soft voting.
        device (torch.device): Il dispositivo su cui eseguire la predizione (CPU o GPU).
        true_class (int, optional): La classe corretta dell'immagine, se conosciuta.

    Returns:
        int: La classe prevista dall'ensemble.
    """
    logging.info(f"Caricamento dell'immagine da interpretare: {image_path}")
    image = Image.open(image_path)
    image = data_transforms(image)

    def predict_fn(images):
        """
        Funzione di predizione usata da LIME.

        Args:
            images (list of np.array): Lista di immagini da predire.

        Returns:
            np.array: Predizioni per le immagini fornite.
        """
        images = [np.uint8(255 * img) if img.dtype == np.float32 else img for img in images]
        images = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0)

        if len(images.shape) == 3:
            images = images.unsqueeze(0)  # Aggiungi la dimensione batch se mancante

        predictions = predict_ensemble_lime(models, images, device, model_weights)
        return predictions

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image.permute(1, 2, 0)),
                                             predict_fn,
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    predicted_class = explanation.top_labels[0]
    logging.info(f"Classe predetta: {predicted_class}")
    if true_class is not None:
        logging.info(f"Classe corretta: {true_class}")

    # Ottieni la spiegazione e la maschera
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10,
                                                hide_rest=False)

    # Converti l'immagine in RGB se è in scala di grigi
    if len(temp.shape) == 2:
        temp = gray2rgb(temp)

    # Visualizza l'immagine originale con i bordi dei superpixel
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.imshow(mask, cmap='jet', alpha=0.5)  # Sovrapponi la maschera con la mappa di colori 'jet'
    plt.title(f'LIME Explanation for class {predicted_class}')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return predicted_class


def compare_explanations(image_path, models, model_weights, device, output_path="comparison_plot.png"):
    """
    Confronta la spiegazione personalizzata con quella generata dalla libreria LIME.

    Args:
        image_path (str): Il percorso dell'immagine da interpretare.
        models (list): Lista di modelli ensemble da usare per la predizione.
        model_weights (list or None): Pesi dei modelli per il soft voting.
        device (torch.device): Il dispositivo su cui eseguire la predizione (CPU o GPU).
        output_path (str): Il percorso in cui salvare il grafico comparativo.

    Returns:
        int: La classe prevista dall'ensemble.
    """
    logging.info(f"Caricamento dell'immagine da interpretare: {image_path}")
    image = Image.open(image_path)
    image = np.array(image)

    if len(image.shape) == 2:
        image = gray2rgb(image)  # Converti in RGB se l'immagine è in scala di grigi

    # Spiegazione personalizzata (custom)
    segments = slic(image, n_segments=150, compactness=10)
    base_image_tensor = data_transforms(Image.fromarray(image)).unsqueeze(0).to(device)
    base_prediction = predict_ensemble_lime(models, base_image_tensor, device, model_weights)
    top_label = np.argmax(base_prediction[0])

    # Perturba l'immagine
    perturbed_images = perturb_image(image, segments)
    perturbed_images_tensor = torch.cat(
        [data_transforms(Image.fromarray(img)).unsqueeze(0) for img in perturbed_images], dim=0).to(device)
    predictions = predict_ensemble_lime(models, perturbed_images_tensor, device, model_weights)

    # Genera la spiegazione personalizzata
    weights = np.mean(predictions[:, top_label], axis=0)
    explanation_custom = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        explanation_custom[segments == i] = weights

    # Spiegazione usando la libreria LIME
    def predict_fn(images):
        images = [np.uint8(255 * img) if img.dtype == np.float32 else img for img in images]
        images = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0)
        predictions = predict_ensemble_lime(models, images, device, model_weights)
        return predictions

    explainer = lime_image.LimeImageExplainer()
    explanation_lime = explainer.explain_instance(np.array(image), predict_fn, top_labels=5, hide_color=0,
                                                  num_samples=1000)
    temp_lime, mask_lime = explanation_lime.get_image_and_mask(top_label, positive_only=True, num_features=10,
                                                               hide_rest=False)

    # Converti temp_lime in RGB se è in scala di grigi
    if len(temp_lime.shape) == 2:
        temp_lime = gray2rgb(temp_lime)

    # Creazione del grafico comparativo
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Immagine originale
    axs[0].imshow(image / 255.0)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    # Spiegazione personalizzata
    axs[1].imshow(mark_boundaries(image / 255.0, segments))
    axs[1].imshow(explanation_custom, alpha=0.5, cmap='jet')
    axs[1].set_title('Custom Explanation')
    axs[1].axis('off')

    # Spiegazione della libreria LIME
    axs[2].imshow(mark_boundaries(temp_lime / 255.0, mask_lime))
    axs[2].imshow(mask_lime, cmap='jet', alpha=0.5)
    axs[2].set_title('LIME Library Explanation')
    axs[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)  # Salva il grafico in un file
    logging.info(f"Comparison plot saved to {output_path}")
    plt.show()

    return top_label


@hydra.main(config_path="./configs", config_name="config_lime")
def main(cfg: DictConfig):
    """
    Funzione principale che gestisce il caricamento del modello e l'esecuzione della spiegazione richiesta.

    Args:
        cfg (DictConfig): Configurazione fornita da Hydra.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Caricamento del modello ensemble da {cfg.ensemble_model_path}...")
    models, model_weights = load_ensemble_model(cfg.ensemble_model_path, device, cfg.num_classes)

    if cfg.explanation_method == "compare":
        # Confronta le spiegazioni
        compare_explanations(cfg.image_path, models, model_weights, device)
    elif cfg.explanation_method == "custom":
        # Usa la spiegazione personalizzata
        custom_explanation(cfg.image_path, models, model_weights, device, true_class=cfg.true_class)
    elif cfg.explanation_method == "lime":
        # Usa la libreria LIME per la spiegazione
        lime_library_explanation(cfg.image_path, models, model_weights, device,
                                 true_class=cfg.true_class)
    else:
        raise ValueError(
            f"Explanation method {cfg.explanation_method} not recognized. Choose 'custom', 'lime', or 'compare'.")


if __name__ == "__main__":
    main()
