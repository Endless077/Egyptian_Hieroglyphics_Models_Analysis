import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from dataset import GlyphData
from typing import Dict
import seaborn as sns


def analyze_dataset(data_dir: str, class_to_idx: Dict[str, int]):
    # Trasformazione di base per caricare le immagini
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Caricamento del dataset
    dataset = GlyphData(class_to_idx=class_to_idx, root=data_dir, transform=transform)

    # Conteggio delle classi
    class_counts = Counter([dataset.classes[idx] for _, idx in dataset.samples])

    # Ordina i conteggi per una migliore visualizzazione
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Visualizzazione della distribuzione delle classi
    plt.figure(figsize=(14, 8))
    plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    plt.xticks(rotation=90)

    # Riduci il numero di etichette visualizzate
    ax = plt.gca()
    ax.set_xticks([i for i, _ in enumerate(sorted_class_counts.keys()) if i % 5 == 0])
    ax.set_xticklabels([label for i, label in enumerate(sorted_class_counts.keys()) if i % 5 == 0])

    plt.xlabel('Classi')
    plt.ylabel('Conteggio')
    plt.title('Distribuzione delle Classi nel Dataset')
    plt.tight_layout()
    plt.show()

    # Statistiche di base
    total_samples = len(dataset)
    num_classes = len(class_counts)
    print(f"Numero totale di campioni: {total_samples}")
    print(f"Numero totale di classi: {num_classes}")


    # Visualizzazione di alcune immagini di esempio per ciascuna classe
    plt.figure(figsize=(20, 20))
    for i, (img_path, label) in enumerate(dataset.samples[:25]):  # Mostra solo le prime 25 immagini
        img = Image.open(img_path)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(dataset.classes[label], fontsize=14)  # Aumenta la dimensione del font
        plt.axis('off')
    plt.suptitle('Esempi di Immagini nel Dataset', fontsize=20)
    plt.show()



    # Analisi delle etichette sconosciute
    unknown_samples = [img_path for img_path, label in dataset.samples if dataset.classes[label] == 'UNKNOWN']
    print(f"Numero di etichette 'UNKNOWN': {len(unknown_samples)}")
    if len(unknown_samples) > 0:
        plt.figure(figsize=(20, 20))
        for i, img_path in enumerate(unknown_samples[:25]):  # Mostra solo le prime 25 immagini sconosciute
            img = Image.open(img_path)
            plt.subplot(5, 5, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title('UNKNOWN')
            plt.axis('off')
        plt.suptitle('Esempi di Immagini con Etichetta "UNKNOWN"')
        plt.show()


if __name__ == "__main__":
    # Percorso del dataset
    data_dir = "prepared_data/train"  # Modifica questo percorso in base alla tua struttura di directory

    # Definizione delle etichette di classe
    class_to_idx = {
        'A55': 0, 'Aa15': 1, 'Aa26': 2, 'Aa27': 3, 'Aa28': 4, 'D1': 5, 'D10': 6, 'D156': 7, 'D19': 8,
        'D2': 9, 'D21': 10, 'D28': 11, 'D34': 12, 'D35': 13, 'D36': 14, 'D39': 15, 'D4': 16, 'D46': 17,
        'D52': 18, 'D53': 19, 'D54': 20, 'D56': 21, 'D58': 22, 'D60': 23, 'D62': 24, 'E1': 25, 'E17': 26,
        'E23': 27, 'E34': 28, 'E9': 29, 'F12': 30, 'F13': 31, 'F16': 32, 'F18': 33, 'F21': 34, 'F22': 35,
        'F23': 36, 'F26': 37, 'F29': 38, 'F30': 39, 'F31': 40, 'F32': 41, 'F34': 42, 'F35': 43, 'F4': 44,
        'F40': 45, 'F9': 46, 'G1': 47, 'G10': 48, 'G14': 49, 'G17': 50, 'G21': 51, 'G25': 52, 'G26': 53,
        'G29': 54, 'G35': 55, 'G36': 56, 'G37': 57, 'G39': 58, 'G4': 59, 'G40': 60, 'G43': 61, 'G5': 62,
        'G50': 63, 'G7': 64, 'H6': 65, 'I10': 66, 'I5': 67, 'I9': 68, 'L1': 69, 'M1': 70, 'M12': 71,
        'M16': 72, 'M17': 73, 'M18': 74, 'M195': 75, 'M20': 76, 'M23': 77, 'M26': 78, 'M29': 79, 'M3': 80,
        'M4': 81, 'M40': 82, 'M41': 83, 'M42': 84, 'M44': 85, 'M8': 86, 'N1': 87, 'N14': 88, 'N16': 89,
        'N17': 90, 'N18': 91, 'N19': 92, 'N2': 93, 'N24': 94, 'N25': 95, 'N26': 96, 'N29': 97, 'N30': 98,
        'N31': 99, 'N35': 100, 'N36': 101, 'N37': 102, 'N41': 103, 'N5': 104, 'O1': 105, 'O11': 106,
        'O28': 107, 'O29': 108, 'O31': 109, 'O34': 110, 'O4': 111, 'O49': 112, 'O50': 113, 'O51': 114,
        'P1': 115, 'P13': 116, 'P6': 117, 'P8': 118, 'P98': 119, 'Q1': 120, 'Q3': 121, 'Q7': 122, 'R4': 123,
        'R8': 124, 'S24': 125, 'S28': 126, 'S29': 127, 'S34': 128, 'S42': 129, 'T14': 130, 'T20': 131,
        'T21': 132, 'T22': 133, 'T28': 134, 'T30': 135, 'U1': 136, 'U15': 137, 'U28': 138, 'U33': 139,
        'U35': 140, 'U7': 141, 'V13': 142, 'V16': 143, 'V22': 144, 'V24': 145, 'V25': 146, 'V28': 147,
        'V30': 148, 'V31': 149, 'V4': 150, 'V6': 151, 'V7': 152, 'W11': 153, 'W14': 154, 'W15': 155,
        'W18': 156, 'W19': 157, 'W22': 158, 'W24': 159, 'W25': 160, 'X1': 161, 'X6': 162, 'X8': 163,
        'Y1': 164, 'Y2': 165, 'Y3': 166, 'Y5': 167, 'Z1': 168, 'Z11': 169, 'Z7': 170
    }

    # Analizza il dataset
    analyze_dataset(data_dir, class_to_idx)
