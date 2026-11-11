# Libraries
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image

from dataset.GlyphDataset.dataset import GlyphData


def create_class_to_idx(dataset_dir: str) -> Dict[str, int]:
    """
    Creates a dictionary that maps each class in the dataset to a numerical index.

    Args:
        dataset_dir (str): Path to the dataset directory.

    Returns:
        Dict[str, int]: A dictionary mapping class names to indices.
    """
    try:
        # List all directories (classes) present in the dataset folder
        classes = sorted([d.name for d in os.scandir(dataset_dir) if d.is_dir()])
    except Exception as e:
        print(f"Error accessing dataset directory: {e}")
        return {}

    # Create a dictionary assigning an index to each class
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return class_to_idx


def display_images(images: List[Tuple[str, int]], title: str, num_images: int = 25) -> None:
    """
    Displays a grid of images.

    Args:
        images (List[Tuple[str, int]]): List of tuples containing image paths and labels.
        title (str): Title for the displayed images.
        num_images (int): Number of images to display (default: 25).
    """
    plt.figure(figsize=(20, 20))
    for i, (img_path, label) in enumerate(images[:num_images]):
        img = Image.open(img_path)
        plt.subplot(5, 5, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(label, fontsize=14)  # Increase the font size of the title
        plt.axis('off')
    plt.suptitle(title, fontsize=20)
    plt.show()


def analyze_dataset(data_dir: str, class_to_idx: Dict[str, int]) -> None:
    """
    Analyzes the dataset for class distribution, visualizes sample images,
    checks for 'UNKNOWN' labels, and analyzes image sizes and pixel intensities.

    Args:
        data_dir (str): Path to the dataset directory.
        class_to_idx (Dict[str, int]): Dictionary mapping class names to their respective indices.
    """
    # Define a basic transformation to load images in grayscale and convert them to tensors
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    # Load the dataset using the GlyphData class and apply the transformation
    dataset = GlyphData(class_to_idx=class_to_idx, root=data_dir, transform=transform)

    # Check if there are any images in the dataset
    if len(dataset.samples) == 0:
        print("No images found in the dataset for the specified classes.")
        return

    # Count the number of images per class
    class_counts = Counter([dataset.classes[idx] for _, idx in dataset.samples])

    # Check if there are any classes in the dictionary without associated images
    for cls in class_to_idx:
        if cls not in class_counts:
            print(f"Class '{cls}' has no images in the dataset.")

    # Sort class counts in descending order for better visualization
    sorted_class_counts = dict(sorted(class_counts.items(), key=lambda item: item[1], reverse=True))

    # Display a bar chart of the class distribution
    plt.figure(figsize=(14, 8))
    plt.bar(sorted_class_counts.keys(), sorted_class_counts.values())
    plt.xticks(rotation=90)

    # Reduce the number of labels on the x-axis for clearer visualization
    ax = plt.gca()
    ax.set_xticks([i for i, _ in enumerate(sorted_class_counts.keys()) if i % 5 == 0])
    ax.set_xticklabels([label for i, label in enumerate(sorted_class_counts.keys()) if i % 5 == 0])

    # Add axis labels and a title to the chart
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Class Distribution in the Dataset')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # Print basic statistics about the dataset
    total_samples = len(dataset)
    num_classes = len(class_counts)
    print(f"Total number of samples: {total_samples}")
    print(f"Total number of classes: {num_classes}")

    # Display some random sample images from the dataset
    random.shuffle(dataset.samples)
    display_images([(img_path, dataset.classes[label]) for img_path, label in dataset.samples], 
                   'Random Examples of Images in the Dataset')

    # Analyze and display images with the 'UNKNOWN' label
    unknown_samples = [(img_path, 'UNKNOWN') for img_path, label in dataset.samples if dataset.classes[label] == 'UNKNOWN']
    print(f"Number of 'UNKNOWN' labels: {len(unknown_samples)}")
    if unknown_samples:
        display_images(unknown_samples, 'Examples of Images with "UNKNOWN" Label')

    # Analyze the distribution of image dimensions (width and height)
    image_sizes = [Image.open(img_path).size for img_path, _ in dataset.samples]
    widths, heights = zip(*image_sizes)
    plt.figure(figsize=(14, 8))
    plt.hist(widths, bins=30, alpha=0.5, label='Width', color='blue')
    plt.hist(heights, bins=30, alpha=0.5, label='Height', color='orange')
    plt.xlabel('Dimensions')
    plt.ylabel('Count')
    plt.title('Image Size Distribution')
    plt.legend()
    plt.grid(axis='y')
    plt.show()

    # Analyze the distribution of pixel intensities in the images
    pixel_intensities = []
    for img_path, _ in dataset.samples:
        # Convert the image to grayscale
        img = Image.open(img_path).convert('L')
        pixel_intensities.extend(list(img.getdata()))
    plt.figure(figsize=(14, 8))
    plt.hist(pixel_intensities, bins=256, range=(0, 256), density=True, color='gray', alpha=0.75)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Distribution')
    plt.grid(axis='y')
    plt.show()

    # Analyze and display the distribution of rare classes (less than 5 samples)
    rare_classes = {k: v for k, v in sorted_class_counts.items() if v < 5}
    print(f"Rare classes (less than 5 samples): {len(rare_classes)}")
    plt.figure(figsize=(14, 8))
    plt.bar(rare_classes.keys(), rare_classes.values(), color='red')
    plt.xticks(rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.title('Distribution of Rare Classes')
    plt.grid(axis='y')
    plt.show()


if __name__ == "__main__":
    # Dataset path
    data_dir = "../datasets/balanced_data/train"

    # Analyze the dataset
    class_mapping = create_class_to_idx(data_dir)
    analyze_dataset(data_dir, class_mapping)
