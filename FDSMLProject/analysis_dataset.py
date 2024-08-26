import os
from collections import Counter
import random

import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from dataset import GlyphData
from typing import Dict

from utils import class_to_idx


def create_class_to_idx(dataset_dir):
    """
    Crea un dizionario che mappa ogni classe nel dataset a un indice numerico.

    Args:
        dataset_dir (str): Il percorso alla directory del dataset.

    Returns:
        dict: Un dizionario che mappa il nome della classe a un indice.
    """
    # Elenca tutte le directory (classi) presenti nella cartella del dataset
    classes = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])

    # Crea un dizionario che assegna un indice a ogni classe
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return class_to_idx


def analyze_dataset(data_dir: str, class_to_idx: Dict[str, int]):
    """
    Analizza il dataset per la distribuzione delle classi, visualizza esempi di immagini,
    verifica la presenza di etichette 'UNKNOWN' e analizza le dimensioni e le intensità delle immagini.

    Args:
        data_dir (str): Il percorso alla directory del dataset.
        class_to_idx (Dict[str, int]): Il dizionario che mappa i nomi delle classi ai rispettivi indici.
    """

    # Definisce una trasformazione di base per caricare le immagini in scala di grigi e convertirle in tensori
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Carica il dataset utilizzando la classe GlyphData e applica la trasformazione
    dataset = GlyphData(class_to_idx=class_to_idx, root=data_dir, transform=transform)

    # Verifica se ci sono immagini nel dataset
    if len(dataset.samples) == 0:
        print("Nessuna immagine trovata nel dataset per le classi specificate.")
        return

    # Conteggia il numero di immagini per ciascuna classe
    class_counts = Counter([dataset.classes[idx] for _, idx in dataset.samples])

    # Controlla se ci sono classi nel dizionario che non hanno immagini associate
    for cls in class_to_idx:
        if cls not in class_counts:
            print(f"Classe '{cls}' non ha immagini nel dataset.")

    # Ordina i conteggi delle classi in ordine decrescente per una migliore visualizzazione
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Visualizza un grafico a barre della distribuzione delle classi
    plt.figure(figsize=(14, 8))
    plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    plt.xticks(rotation=90)

    # Riduce il numero di etichette sull'asse x per una visualizzazione più chiara
    ax = plt.gca()
    ax.set_xticks([i for i, _ in enumerate(sorted_class_counts.keys()) if i % 5 == 0])
    ax.set_xticklabels([label for i, label in enumerate(sorted_class_counts.keys()) if i % 5 == 0])

    # Aggiunge etichette agli assi e un titolo al grafico
    plt.xlabel('Classi')
    plt.ylabel('Conteggio')
    plt.title('Distribuzione delle Classi nel Dataset')
    plt.tight_layout()
    plt.show()

    # Stampa alcune statistiche di base sul dataset
    total_samples = len(dataset)
    num_classes = len(class_counts)
    print(f"Numero totale di campioni: {total_samples}")
    print(f"Numero totale di classi: {num_classes}")

    # Visualizza alcune immagini di esempio prese casualmente dal dataset
    plt.figure(figsize=(20, 20))

    # Mescola l'ordine delle immagini nel dataset per una selezione casuale
    random.shuffle(dataset.samples)

    # Visualizza un sottoinsieme di immagini (25 immagini casuali)
    for i, (img_path, label) in enumerate(dataset.samples[:25]):
        img = Image.open(img_path)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(dataset.classes[label], fontsize=14)  # Aumenta la dimensione del font del titolo
        plt.axis('off')
    plt.suptitle('Esempi Casuali di Immagini nel Dataset', fontsize=20)
    plt.show()

    # Analizza e visualizza immagini con etichetta 'UNKNOWN'
    unknown_samples = [img_path for img_path, label in dataset.samples if dataset.classes[label] == 'UNKNOWN']
    print(f"Numero di etichette 'UNKNOWN': {len(unknown_samples)}")
    if len(unknown_samples) > 0:
        plt.figure(figsize=(20, 20))
        for i, img_path in enumerate(unknown_samples[:25]):  # Mostra solo le prime 25 immagini 'UNKNOWN'
            img = Image.open(img_path)
            plt.subplot(5, 5, i + 1)
            plt.imshow(img, cmap='gray')
            plt.title('UNKNOWN')
            plt.axis('off')
        plt.suptitle('Esempi di Immagini con Etichetta "UNKNOWN"')
        plt.show()

    # Analizza la distribuzione delle dimensioni delle immagini (larghezza e altezza)
    image_sizes = [Image.open(img_path).size for img_path, _ in dataset.samples]
    widths, heights = zip(*image_sizes)
    plt.figure(figsize=(14, 8))
    plt.hist(widths, bins=30, alpha=0.5, label='Larghezza')
    plt.hist(heights, bins=30, alpha=0.5, label='Altezza')
    plt.xlabel('Dimensioni')
    plt.ylabel('Conteggio')
    plt.title('Distribuzione delle Dimensioni delle Immagini')
    plt.legend()
    plt.show()

    # Analizza la distribuzione delle intensità dei pixel nelle immagini
    pixel_intensities = []
    for img_path, _ in dataset.samples:
        img = Image.open(img_path).convert('L')  # Converti l'immagine in scala di grigi
        pixel_intensities.extend(list(img.getdata()))
    plt.figure(figsize=(14, 8))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.xlabel('Intensità dei Pixel')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione delle Intensità dei Pixel')
    plt.show()

    # Analizza e visualizza la distribuzione delle classi rare (meno di 5 campioni)
    rare_classes = {k: v for k, v in sorted_class_counts.items() if v < 5}
    print(f"Classi rare (meno di 5 campioni): {len(rare_classes)}")
    plt.figure(figsize=(14, 8))
    plt.bar(rare_classes.keys(), rare_classes.values(), color='red')
    plt.xticks(rotation=90)
    plt.xlabel('Classi')
    plt.ylabel('Conteggio')
    plt.title('Distribuzione delle Classi Rare')
    plt.show()


if __name__ == "__main__":
    # Percorso del dataset
    data_dir = "balanced_data/train"

    # Analizza il dataset
    analyze_dataset(data_dir, class_to_idx)
