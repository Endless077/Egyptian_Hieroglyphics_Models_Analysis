from collections import Counter
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from dataset import GlyphData
from typing import Dict
from utils import class_to_idx


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

    # Distribuzione delle dimensioni delle immagini
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

    # Statistiche delle intensità dei pixel
    pixel_intensities = []
    for img_path, _ in dataset.samples:
        img = Image.open(img_path).convert('L')
        pixel_intensities.extend(list(img.getdata()))
    plt.figure(figsize=(14, 8))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.xlabel('Intensità dei Pixel')
    plt.ylabel('Frequenza')
    plt.title('Distribuzione delle Intensità dei Pixel')
    plt.show()

    # Analisi delle classi rare
    rare_classes = {k: v for k, v in sorted_class_counts.items() if v < 5}
    print(f"Classi rare (meno di 10 campioni): {len(rare_classes)}")
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
