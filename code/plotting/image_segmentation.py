# Libraries
import cv2
import sys
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.GlyphNet.model import Glyphnet
from dataset.yolo_extraction_dataset import class_to_idx

# Append the path to GlyphNet
sys.path.append('../models/GlyphNet')

# Dictionary that maps class indices to their names
idx_to_class = {v: k for k, v in class_to_idx.items()}

# Load the image
image_path = '../datasets/data/Dataset/Pictures/hieroglyphics-stone-2.jpg'


# Function to load an image safely
def load_image(image_path: str) -> np.ndarray:
    """Loads an image from the specified path and returns it as a NumPy array.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    try:
        image = Image.open(image_path)
        return np.array(image)
    except FileNotFoundError:
        print(f"Error: Image file {image_path} not found.")
        sys.exit(1)

image = load_image(image_path)


# Function to process the image (grayscale, equalize, blur, edges)
def process_image(image: np.ndarray) -> np.ndarray:
    """Processes the image by converting it to grayscale, equalizing, blurring, 
    and finding edges, followed by dilation and erosion.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        np.ndarray: The processed image after applying all transformations.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    edges = cv2.Canny(blurred, 100, 120)
    kernel = np.ones((2, 2), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return cv2.erode(dilated, kernel, iterations=1)

# Process the image to get edges
eroded = process_image(image)

# Find contours in the processed image
contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on size to avoid noise
min_contour_area = 0.01 * (image.shape[0] * image.shape[1])  # 1% of the image area
filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

# Load the trained GlyphNet model for hieroglyph classification
model = Glyphnet(num_classes=171)  # Make sure to replace 171 with the correct number of classes


# Function to load model weights safely
def load_model_weights(model: Glyphnet) -> Glyphnet:
    """Loads the weights for the GlyphNet model from a specified file.

    Args:
        model (Glyphnet): The Glyphnet model instance.

    Returns:
        Glyphnet: The model with loaded weights.

    Raises:
        Exception: If there's an error loading the model weights.
    """
    try:
        model.load_state_dict(torch.load("../results/results_glyphnet/best_weights/best_model_weights.pth"))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

model = load_model_weights(model)

# Set device for model inference
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

# Define the transformation for the image before classification
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


# Function to generate captions (classifications) for each image segment
def generate_captions(image: np.ndarray, 
                      contours: list[np.ndarray], 
                      model: Glyphnet, 
                      transform: transforms.Compose, 
                      device: torch.device, 
                      confidence_threshold: float = 0.5) -> list[tuple[int, float]]:
    """Generates captions for image segments based on model predictions.

    Args:
        image (np.ndarray): The original image as a NumPy array.
        contours (list): List of contours found in the image.
        model (Glyphnet): The trained Glyphnet model.
        transform (torchvision.transforms.Compose): Transformations for image preprocessing.
        device (torch.device): Device to run the model on (CPU or GPU).
        confidence_threshold (float): Minimum confidence for predictions to be included.

    Returns:
        list: A list of tuples containing class indices and their corresponding confidence scores.
    """
    captions = []
    for cnt in contours:
        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(cnt)
        segment = image[y:y + h, x:x + w]
        segment_pil = Image.fromarray(segment)
        segment_tensor = transform(segment_pil).unsqueeze(0).to(device)

        with torch.no_grad():  # Use context manager for no_grad
            output = model(segment_tensor)
            probabilities = torch.softmax(output, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            # Add the classification only if it exceeds the confidence threshold
            if max_prob.item() >= confidence_threshold:
                captions.append((predicted.item(), max_prob.item()))
    return captions

# Generate captions for the image segments with a specific confidence threshold
confidence_threshold = 0.1
captions = generate_captions(image, filtered_contours, model, transform, DEVICE, confidence_threshold)

# Print captions (classifications) for each segment with the respective confidence
for idx, (caption, confidence) in enumerate(captions):
    print(f"Segment {idx + 1}: Class {idx_to_class[caption]} with confidence {confidence:.2f}")


# Function to draw contours on the image
def draw_contours(image: np.ndarray, contours: list[np.ndarray], color: tuple[int, int, int] = (0, 255, 0)) -> None:
    """Draws bounding boxes around contours on the image.

    Args:
        image (np.ndarray): The original image as a NumPy array.
        contours (list): List of contours to be drawn.
        color (tuple): Color of the bounding boxes in BGR format.
    """
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

# Show the original image with all detected contours
contour_image = image.copy()
draw_contours(contour_image, filtered_contours)

plt.figure(figsize=(10, 10))
plt.imshow(contour_image)
plt.axis('off')
plt.show()

# Show the image with only the accepted segments (those that exceed the confidence threshold)
accepted_contour_image = image.copy()
for cnt, (caption, confidence) in zip(filtered_contours, captions):
    if confidence >= confidence_threshold:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(accepted_contour_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.figure(figsize=(10, 10))
plt.imshow(accepted_contour_image)
plt.axis('off')
plt.show()

# Count occurrences of each class in the classified segments
class_counts = {}
for caption, _ in captions:
    class_name = idx_to_class[caption]
    class_counts[class_name] = class_counts.get(class_name, 0) + 1

# Determine the most frequent class (family of hieroglyphs)
most_frequent_class = max(class_counts, key=class_counts.get)
most_frequent_count = class_counts[most_frequent_class]

# Describe the image based on the most frequent family
image_description = f"The most frequent family of hieroglyphs is '{most_frequent_class}' with {most_frequent_count} occurrences."

# Print the image description
print(image_description)
