import numpy as np
import timm
import torch
import torchvision.transforms as transforms
from skimage.segmentation import slic
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from PIL import Image
import os

from ATCNet import ATCNet
from model import Glyphnet

# Imposta il dispositivo su GPU se disponibile, altrimenti usa la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_name):
    if model_name == "Glyphnet":
        model = Glyphnet()
        checkpoint = torch.load("results/2024-08-09_11-24-30/best_model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    elif model_name == "ATCNet":
        model = ATCNet(n_classes=171)  # Specifica il numero di classi corretto per il tuo modello
        checkpoint = torch.load("result_atcnet/2024-08-09_17-49-34/best_model_weights.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    elif model_name == "tresnet_m":
        model = timm.create_model('tresnet_m', pretrained=True, num_classes=50)
        checkpoint = torch.load("best_tresnet_model.pth", map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    model.eval()  # Imposta il modello in modalità valutazione
    model.to(device)  # Sposta il modello sul dispositivo (GPU o CPU)
    return model


# Funzione di predizione generica
def predict(input_tensor, model):
    model.eval()
    with torch.no_grad():  # Disabilita il calcolo del gradiente per risparmiare memoria
        input_tensor = input_tensor.to(device)  # Sposta il tensore sul dispositivo (GPU o CPU)
        output = model(input_tensor)  # Ottieni l'output del modello
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Calcola le probabilità usando softmax
        return probabilities.cpu().numpy()  # Converti le probabilità in un array numpy e spostale sulla CPU


def preprocess_image(image, model_name):
    if model_name == "Glyphnet":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Converte l'immagine in scala di grigi
            transforms.Resize((64, 64)),  # Dimensioni specifiche per Glyphnet (adatta le dimensioni secondo necessità)
            transforms.ToTensor(),  # Converte l'immagine in un tensore
        ])
    elif model_name == "ATCNet":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Converte l'immagine in scala di grigi
            transforms.Resize((128, 128)),  # Dimensioni specifiche per ATCNet (adatta le dimensioni secondo necessità)
            transforms.ToTensor(),  # Converte l'immagine in un tensore
        ])
    elif model_name == "tresnet_m":
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # Converte l'immagine in 3 canali per tresnet_m
            transforms.Resize((224, 224)),  # Ridimensiona l'immagine a 224x224
            transforms.ToTensor(),  # Converte l'immagine in un tensore
        ])
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return transform(image).unsqueeze(0)  # Aggiunge una dimensione per il batch


# Funzione per creare perturbazioni dell'immagine
def perturb_image(image, segments):
    perturbed_images = []
    num_segments = np.max(segments) + 1  # Ottieni il numero di segmenti

    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i  # Crea una maschera per il segmento corrente
        if np.any(mask):  # Verifica che il segmento non sia vuoto
            mean_value = np.mean(image[mask], axis=0)
            std_value = np.std(image[mask], axis=0)
            if not np.isnan(mean_value).any() and not np.isnan(
                    std_value).any():  # Assicurati che la media e la deviazione standard non siano NaN
                perturbed_image[mask] = np.random.normal(loc=mean_value,
                                                         scale=std_value)  # Sostituisci i pixel del segmento con rumore
            else:
                perturbed_image[mask] = np.mean(image[mask],
                                                axis=0)  # Usa la media semplice se ci sono problemi con il calcolo della deviazione standard
        perturbed_images.append(perturbed_image)  # Aggiungi l'immagine perturbata alla lista

    return np.array(perturbed_images)  # Restituisci le immagini perturbate come array numpy


# Funzione per spiegare un'immagine
def explain_image(image, model, model_name):
    # Converti l'immagine in scala di grigi se necessario
    image = np.array(image)
    if len(image.shape) == 2:
        image = gray2rgb(image)  # Converte in RGB se è in scala di grigi

    # Segmenta l'immagine in superpixel
    segments = slic(image, n_segments=150, compactness=10)  # Aumenta il numero di segmenti

    # Fai la previsione base
    base_image_tensor = preprocess_image(Image.fromarray(image), model_name).float().to(
        device)  # Preprocessa e sposta sul dispositivo
    base_prediction = predict(base_image_tensor, model)  # Ottieni la previsione di base

    # Crea perturbazioni dell'immagine
    perturbed_images = perturb_image(image, segments)

    # Preprocessa le immagini perturbate e crea il batch tensor
    perturbed_images_tensor = torch.cat([preprocess_image(Image.fromarray(img), model_name).float() for img in perturbed_images],
                                        dim=0).to(device)

    # Fai previsioni sulle immagini perturbate
    predictions = predict(perturbed_images_tensor, model)

    # Analizza le previsioni
    top_label = np.argmax(base_prediction[0])  # Ottieni l'etichetta di previsione principale
    weights = np.mean(predictions[:, top_label], axis=0)  # Calcola il peso medio per l'etichetta principale

    # Genera la spiegazione
    explanation = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        mask = segments == i
        explanation[mask] = weights  # Assegna il peso calcolato al segmento

    # Visualizza la spiegazione sovrapponendola all'immagine originale
    plt.figure(figsize=(8, 6))
    plt.imshow(mark_boundaries(image / 255.0, segments))
    plt.imshow(explanation, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.title(f"LIME Explanation for {model.__class__.__name__}")
    plt.show()


# Funzione principale per eseguire LIME su un modello specifico
def main_lime(model_name):
    model = load_model(model_name)  # Carica il modello specificato

    if model_name == 'Glyphnet' or model_name == 'ATCNet':
        image_paths = [
            "balanced_data/train/Aa26/aug_4_3534f1a21ff6b826a1268c3ae2e13d23.png",
            "balanced_data/train/D1/5c6f10aadc08904fa1edbff37c6da96d.png",
            "balanced_data/train/D1/d8bfe00858c74d3b3a642434917e3abd.png"
        ]
    else:
        image_paths = [
            "classification_dataset/train/4/Screen-Shot-2020-07-06-at-4-52-56-PM_1_png.rf"
            ".d4a00cb87156c556560216c84e118b50_516_341.jpg",
            "classification_dataset/train/49/wall_section9237_3_png.rf.1d0ca3489d53ac9e8ef34f2bcf64a4ac_321_400.jpg",
            "classification_dataset/train/28/Screen-Shot-2020-07-06-at-4-13-36-PM_0_png.rf"
            ".c3f68932bc7cccaa4b9336b7282b83ed_240_347.jpg"
        ]

    # Itera su ciascun percorso di immagine
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)  # Carica l'immagine
        explain_image(image, model, model_name)  # Spiega l'immagine


# Esempio di utilizzo
if __name__ == "__main__":
    main_lime("Glyphnet")
    main_lime("ATCNet")
    main_lime("tresnet_m")
