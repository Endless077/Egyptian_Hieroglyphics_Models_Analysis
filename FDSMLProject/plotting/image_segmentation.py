import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import Glyphnet
import utils

# Dizionario che mappa gli indici delle classi ai loro nomi
idx_to_class = {v: k for k, v in utils.class_to_idx.items()}

# Carica l'immagine
image_path = '../datasets/data/Dataset/Pictures/hieroglyphics-stone-2.jpg'
image = Image.open(image_path)
image = np.array(image)

# Converti l'immagine in scala di grigi per facilitare l'elaborazione
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Migliora il contrasto dell'immagine con l'equalizzazione dell'istogramma
equalized = cv2.equalizeHist(gray)

# Riduci il rumore nell'immagine con un filtro Gaussiano
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Applica l'algoritmo di rilevamento dei bordi Canny con parametri ottimizzati
edges = cv2.Canny(blurred, 100, 120)

# Applica dilatazione ed erosione per unire i contorni frammentati
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Trova i contorni nell'immagine elaborata
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtra i contorni in base alla dimensione per evitare rumori
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 110]

# Carica il modello Glyphnet addestrato per la classificazione dei geroglifici
model = Glyphnet(num_classes=171)  # Assicurati di sostituire 171 con il numero corretto di classi
model.load_state_dict(torch.load("../results/results:glyphnet/best_weights/best_model_weights.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Definisce la trasformazione per l'immagine prima della classificazione
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

# Funzione per generare le caption (classificazioni) per ogni segmento di immagine
def generate_captions(image, contours, model, transform, device, confidence_threshold=0.5):
    captions = []
    for cnt in contours:
        # Ottieni la bounding box per il contorno
        x, y, w, h = cv2.boundingRect(cnt)
        segment = image[y:y + h, x:x + w]
        segment_pil = Image.fromarray(segment)
        segment_tensor = transform(segment_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(segment_tensor)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            # Aggiungi la classificazione solo se supera la soglia di confidenza
            if max_prob.item() >= confidence_threshold:
                captions.append((predicted.item(), max_prob.item()))
    return captions

# Genera le caption per i segmenti dell'immagine con una soglia di confidenza specifica
confidence_threshold = 0.1
captions = generate_captions(image, filtered_contours, model, transform, device, confidence_threshold)

# Stampa le caption (classificazioni) per ogni segmento con la rispettiva confidenza
for idx, (caption, confidence) in enumerate(captions):
    print(f"Segment {idx + 1}: Class {idx_to_class[caption]} with confidence {confidence:.2f}")

# Mostra l'immagine originale con tutti i contorni rilevati
contour_image = image.copy()
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(contour_image)
plt.axis('off')
plt.show()

# Mostra l'immagine con solo i segmenti accettati (quelli che superano la soglia di confidenza)
accepted_contour_image = image.copy()
for cnt, (caption, confidence) in zip(filtered_contours, captions):
    if confidence >= confidence_threshold:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(accepted_contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(accepted_contour_image)
plt.axis('off')
plt.show()

# Conta le occorrenze di ciascuna classe nei segmenti classificati
class_counts = {}
for caption, _ in captions:
    class_name = idx_to_class[caption]
    if class_name not in class_counts:
        class_counts[class_name] = 1
    else:
        class_counts[class_name] += 1

# Determina la classe (famiglia di geroglifici) più numerosa
most_frequent_class = max(class_counts, key=class_counts.get)
most_frequent_count = class_counts[most_frequent_class]

# Descrivi l'immagine basata sulla famiglia più numerosa
image_description = f"La famiglia di geroglifici più numerosa è '{most_frequent_class}' con {most_frequent_count} occorrenze."

# Stampa la descrizione dell'immagine
print(image_description)
