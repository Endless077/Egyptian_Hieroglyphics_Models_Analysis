import numpy as np
import torch
import torchvision.transforms as transforms
from skimage.segmentation import slic
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from PIL import Image
import os

from model import Glyphnet

# Imposta il dispositivo su GPU se disponibile, altrimenti usa la CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carica il modello con i pesi addestrati
model = Glyphnet()
checkpoint = torch.load("results/2024-07-31_11-57-19/best_model_weights.pth", map_location=device)

# Carica i pesi nel modello
model.load_state_dict(checkpoint, strict=False)
model.eval()  # Imposta il modello in modalità valutazione
model.to(device)  # Sposta il modello sul dispositivo (GPU o CPU)


# Definisci la funzione di predizione
def predict(input_tensor):
    model.eval()
    with torch.no_grad():  # Disabilita il calcolo del gradiente per risparmiare memoria
        input_tensor = input_tensor.to(device)  # Sposta il tensore sul dispositivo (GPU o CPU)
        output = model(input_tensor)  # Ottieni l'output del modello
        probabilities = torch.nn.functional.softmax(output, dim=1)  # Calcola le probabilità usando softmax
        return probabilities.cpu().numpy()  # Converti le probabilità in un array numpy e spostale sulla CPU


# Definisci la funzione di preprocessing per le immagini
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Converte l'immagine in scala di grigi
        transforms.ToTensor(),  # Converte l'immagine in un tensore
    ])
    return transform(image).unsqueeze(0)  # Aggiunge una dimensione per il batch


# Definisci la funzione per creare perturbazioni dell'immagine
def perturb_image(image, segments, base_prediction, threshold=0.05):
    perturbed_images = []
    num_segments = np.max(segments) + 1  # Ottieni il numero di segmenti

    # Crea perturbazioni per ciascun segmento
    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i  # Crea una maschera per il segmento corrente
        if np.any(mask):  # Controlla se il segmento contiene pixel
            perturbed_image[mask] = np.mean(image[mask], axis=0)  # Sostituisci i pixel del segmento con la media
            perturbed_images.append(perturbed_image)  # Aggiungi l'immagine perturbata alla lista

    return np.array(perturbed_images)  # Restituisci le immagini perturbate come array numpy


# Definisci la funzione per spiegare un'immagine
def explain_image(image):
    # Converti l'immagine in scala di grigi se necessario
    image = np.array(image)
    if len(image.shape) == 2:
        image = gray2rgb(image)  # Converte in RGB se è in scala di grigi

    # Segmenta l'immagine in superpixel
    segments = slic(image, n_segments=100, compactness=10)  # Aumenta il numero di segmenti

    # Fai la previsione base
    base_image_tensor = preprocess_image(Image.fromarray(image)).float().to(
        device)  # Preprocessa e sposta sul dispositivo
    base_prediction = predict(base_image_tensor)  # Ottieni la previsione di base

    # Crea perturbazioni dell'immagine
    perturbed_images = perturb_image(image, segments, base_prediction)

    # Preprocessa le immagini perturbate e crea il batch tensor
    perturbed_images_tensor = torch.cat([preprocess_image(Image.fromarray(img)).float() for img in perturbed_images],
                                        dim=0).to(device)

    # Fai previsioni sulle immagini perturbate
    predictions = predict(perturbed_images_tensor)

    # Analizza le previsioni
    top_label = np.argmax(base_prediction[0])  # Ottieni l'etichetta di previsione principale
    weights = np.mean(predictions[:, top_label], axis=0)  # Calcola il peso medio per l'etichetta principale

    # Genera la spiegazione
    explanation = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        mask = segments == i
        explanation[mask] = weights  # Assegna il peso calcolato al segmento

    # Visualizza la spiegazione sovrapponendola all'immagine originale
    plt.imshow(mark_boundaries(image / 255.0, segments))
    plt.imshow(explanation, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.show()


# Definizione dei label di training
train_path = "prepared_data/train"
train_labels = {l: i for i, l in enumerate(
    sorted([p.strip("/") for p in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, p))]))}

# Esempio di utilizzo
if __name__ == "__main__":
    # Lista di percorsi delle immagini da elaborare
    image_paths = [
        "data/Dataset/Automated/Preprocessed/3/030000_D35.png",
        "data/Dataset/Automated/Preprocessed/3/030001_V28.png",
        "data/Dataset/Automated/Preprocessed/3/030004_D58.png"
    ]

    # Itera su ciascun percorso di immagine
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        image = Image.open(image_path)  # Carica l'immagine
        explain_image(image)  # Spiega l'immagine
