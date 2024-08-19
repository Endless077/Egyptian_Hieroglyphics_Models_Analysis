import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from model import Glyphnet
import utils


idx_to_class = {v: k for k, v in utils.class_to_idx.items()}

# Carica l'immagine
image_path = 'dataset/train/images/Screen-Shot-2020-07-06-at-4-06-21-PM_2_png.rf.3c33d4907c2df4907d46deb8aae38e71.jpg'
image = Image.open(image_path)
image = np.array(image)

# Converti l'immagine in scala di grigi
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Migliora il contrasto con l'equalizzazione dell'istogramma
equalized = cv2.equalizeHist(gray)

# Riduci il rumore con il filtro Gaussian
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Usa il Canny Edge Detector con parametri ottimizzati
edges = cv2.Canny(blurred, 100, 120)

# Applicare dilatazione ed erosione per unire i contorni
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)
eroded = cv2.erode(dilated, kernel, iterations=1)

# Trova i contorni
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtra i contorni in base alla dimensione
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 110]

# Carica il modello addestrato
model = Glyphnet(num_classes=171)  # Sostituisci 171 con il numero corretto di classi
model.load_state_dict(torch.load("results/2024-08-09_11-24-30/best_model_weights.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Trasformazione delle immagini
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# Funzione per generare le caption per ogni segmento
def generate_captions(image, contours, model, transform, device, confidence_threshold=0.5):
    captions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        segment = image[y:y + h, x:x + w]
        segment_pil = Image.fromarray(segment)
        segment_tensor = transform(segment_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(segment_tensor)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            if max_prob.item() >= confidence_threshold:
                captions.append((predicted.item(), max_prob.item()))
    return captions


# Genera le caption per i segmenti con soglia di confidenza
confidence_threshold = 0.1
captions = generate_captions(image, filtered_contours, model, transform, device, confidence_threshold)

# Stampa le caption in console con la confidenza
for idx, (caption, confidence) in enumerate(captions):
    print(f"Segment {idx + 1}: Class {idx_to_class[caption]} with confidence {confidence:.2f}")

# Mostra l'immagine segmentata
contour_image = image.copy()
for cnt in filtered_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(contour_image)
plt.axis('off')
plt.show()

# Mostra l'immagine segmentata solo con i segmenti accettati
accepted_contour_image = image.copy()
for cnt, (caption, confidence) in zip(filtered_contours, captions):
    if confidence >= confidence_threshold:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(accepted_contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(accepted_contour_image)
plt.axis('off')
plt.show()

# Conta le occorrenze di ciascuna classe
class_counts = {}
for caption, _ in captions:
    class_name = idx_to_class[caption]
    if class_name not in class_counts:
        class_counts[class_name] = 1
    else:
        class_counts[class_name] += 1

# Determina la famiglia più numerosa
most_frequent_class = max(class_counts, key=class_counts.get)
most_frequent_count = class_counts[most_frequent_class]

# Descrivi l'immagine basata sulla famiglia più numerosa
image_description = f"La famiglia di geroglifici più numerosa è '{most_frequent_class}' con {most_frequent_count} occorrenze."

# Stampa la descrizione dell'immagine
print(image_description)





