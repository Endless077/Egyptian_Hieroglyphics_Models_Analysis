import sys
import torch
import numpy as np
import timm
import torchvision.transforms as transforms
from skimage.segmentation import slic, mark_boundaries
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from PIL import Image
from lime import lime_image  # Import the LIME library


sys.path.append('../models/GlyphNet')
from models.ATCNet import ATCNet
from models.GlyphNet.model import Glyphnet

# Imposta il dispositivo su GPU se disponibile, altrimenti usa la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    """
    Carica il modello specificato dal nome e carica i pesi salvati.

    Args:
        model_name (str): Nome del modello da caricare (Glyphnet, ATCNet, tresnet_m).

    Returns:
        model (torch.nn.Module): Il modello caricato.
    """
    if model_name == "Glyphnet":
        model = Glyphnet()
        checkpoint = torch.load("../results/results_glyphnet/best_weights/best_model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    elif model_name == "ATCNet":
        model = ATCNet(n_classes=171)  # Specifica il numero di classi corretto per il tuo modello
        checkpoint = torch.load("../results/results_atcnet/best_weights/best_model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    elif model_name == "tresnet_m":
        model = timm.create_model('tresnet_m', pretrained=True, num_classes=50)
        checkpoint = torch.load("../results/results_tresnet/best_weights/best_model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    model.eval()  # Imposta il modello in modalità valutazione
    model.to(device)  # Sposta il modello sul dispositivo (GPU o CPU)
    return model


def preprocess_image(image, model_name):
    """
    Preprocessa un'immagine in base al modello specificato.

    Args:
        image (PIL.Image): Immagine da preprocessare.
        model_name (str): Nome del modello che richiede la preprocessazione.

    Returns:
        torch.Tensor: Immagine preprocessata come tensore.
    """
    transform_dict = {
        "Glyphnet": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]),
        "ATCNet": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ]),
        "tresnet_m": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

    if model_name not in transform_dict:
        raise ValueError(f"Model {model_name} not recognized.")

    transform = transform_dict[model_name]
    return transform(image).unsqueeze(0)  # Aggiunge una dimensione per il batch


def predict(input_tensor, model):
    """
    Effettua una predizione usando il modello fornito.

    Args:
        input_tensor (torch.Tensor): Input preprocessato.
        model (torch.nn.Module): Modello da utilizzare per la predizione.

    Returns:
        np.array: Probabilità previste per ciascuna classe.
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities.cpu().numpy()


def perturb_image(image, segments):
    """
    Crea versioni perturbate dell'immagine basate sui segmenti.

    Args:
        image (np.array): Immagine da perturbare.
        segments (np.array): Segmenti creati dall'algoritmo SLIC.

    Returns:
        np.array: Immagini perturbate.
    """
    perturbed_images = []
    num_segments = np.max(segments) + 1

    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i
        if np.any(mask):
            mean_value = np.mean(image[mask], axis=0)
            std_value = np.std(image[mask], axis=0)
            if not np.isnan(mean_value).any() and not np.isnan(std_value).any():
                perturbed_image[mask] = np.random.normal(loc=mean_value, scale=std_value)
            else:
                perturbed_image[mask] = mean_value
        perturbed_images.append(perturbed_image)

    return np.array(perturbed_images)


def explain_image_custom(image, model, model_name):
    """
    Genera una spiegazione personalizzata per un'immagine utilizzando perturbazioni dei segmenti.

    Args:
        image (PIL.Image): Immagine da spiegare.
        model (torch.nn.Module): Modello da utilizzare per la predizione.
        model_name (str): Nome del modello per la preprocessazione.

    Returns:
        tuple: Segmenti dell'immagine e spiegazione generata.
    """
    image = np.array(image)
    if len(image.shape) == 2:
        image = gray2rgb(image)

    segments = slic(image, n_segments=150, compactness=10)

    base_image_tensor = preprocess_image(Image.fromarray(image), model_name).float().to(device)
    base_prediction = predict(base_image_tensor, model)

    perturbed_images = perturb_image(image, segments)
    perturbed_images_tensor = torch.cat(
        [preprocess_image(Image.fromarray(img), model_name).float() for img in perturbed_images], dim=0).to(device)

    predictions = predict(perturbed_images_tensor, model)

    top_label = np.argmax(base_prediction[0])
    weights = np.mean(predictions[:, top_label], axis=0)

    explanation = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        explanation[segments == i] = weights

    return segments, explanation


def explain_image_lime(image, model, model_name):
    """
    Genera una spiegazione per un'immagine utilizzando la libreria LIME.

    Args:
        image (PIL.Image): Immagine da spiegare.
        model (torch.nn.Module): Modello da utilizzare per la predizione.
        model_name (str): Nome del modello per la preprocessazione.

    Returns:
        tuple: Immagine con sovrapposizione della spiegazione e maschera LIME.
    """
    def predict_fn(images):
        images = torch.stack([preprocess_image(Image.fromarray(img), model_name).squeeze(0) for img in images])
        return predict(images, model)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image),
                                             predict_fn,
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    predicted_class = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10,
                                                hide_rest=False)

    return temp, mask


def compare_explanations(image, model, model_name):
    """
    Confronta le spiegazioni generate dal metodo custom e dalla libreria LIME.

    Args:
        image (PIL.Image): Immagine da spiegare.
        model (torch.nn.Module): Modello da utilizzare per la predizione.
        model_name (str): Nome del modello per la preprocessazione.

    Returns:
        None
    """
    original_image = np.array(image)
    if len(original_image.shape) == 2:
        original_image = gray2rgb(original_image)

    # Spiegazione custom
    segments_custom, explanation_custom = explain_image_custom(image, model, model_name)

    # Spiegazione LIME
    temp_lime, mask_lime = explain_image_lime(original_image, model, model_name)

    # Creazione del grafico comparativo
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(original_image / 255.0)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(mark_boundaries(original_image / 255.0, segments_custom))
    axs[1].imshow(explanation_custom, alpha=0.5, cmap='jet')
    axs[1].set_title('Custom Explanation')
    axs[1].axis('off')

    axs[2].imshow(mark_boundaries(temp_lime / 255.0, mask_lime))
    axs[2].imshow(mask_lime, cmap='jet', alpha=0.5)
    axs[2].set_title('LIME Library Explanation')
    axs[2].axis('off')

    plt.show()


def main_lime(model_name):
    """
    Esegue la generazione di spiegazioni per un insieme di immagini utilizzando sia il metodo custom che LIME.

    Args:
        model_name (str): Nome del modello da utilizzare.

    Returns:
        None
    """
    model = load_model(model_name)

    # Definisce i percorsi delle immagini in base al modello
    if model_name in ['Glyphnet', 'ATCNet']:
        image_paths = [
            "../datasets/balanced_data/train/Aa26/aug_4_3534f1a21ff6b826a1268c3ae2e13d23.png",
            "../datasets/balanced_data/train/D1/5c6f10aadc08904fa1edbff37c6da96d.png",
            "../datasets/balanced_data/train/D1/d8bfe00858c74d3b3a642434917e3abd.png"
        ]
    else:
        image_paths = [
            "../datasets/classification_dataset/train/4/Screen-Shot-2020-07-06-at-4-52-56-PM_1_png.rf"
            ".d4a00cb87156c556560216c84e118b50_516_341.jpg",
            "../datasets/classification_dataset/train/49/wall_section9237_3_png.rf"
            ".1d0ca3489d53ac9e8ef34f2bcf64a4ac_321_400.jpg",
            "../datasets/classification_dataset/train/47/ZofUksf_4_png.rf.8c84a343c41dc2cfb082f27ee7004230_469_122.jpg"
        ]

    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)
        compare_explanations(image, model, model_name)


if __name__ == "__main__":
    main_lime("Glyphnet")
    main_lime("ATCNet")
    main_lime("tresnet_m")
