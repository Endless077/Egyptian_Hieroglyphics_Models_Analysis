# Libraries
import sys
import timm

import torch
import numpy as np
from lime import lime_image

from skimage.segmentation import slic, mark_boundaries
import torchvision.transforms as transforms
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from PIL import Image
import logging

# Append the custom model path for loading models
sys.path.append('../models/GlyphNet')
from models.ATCNet import ATCNet
from models.GlyphNet.model import Glyphnet

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
NUM_SEGMENTS = 150
COMPACTNESS = 10
GLYPHNET_INPUT_SIZE = (64, 64)
ATCNET_INPUT_SIZE = (128, 128)
TRESNET_INPUT_SIZE = (224, 224)


def load_model(model_name: str) -> torch.nn.Module:
    """
    Loads the specified model and its saved weights.

    Args:
        model_name (str): Name of the model to load (Glyphnet, ATCNet, tresnet_m).

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model_mapping = {
        "Glyphnet": (Glyphnet, "../results/results_glyphnet/best_weights/best_model_weights.pth"),
        "ATCNet": (lambda: ATCNet(n_classes=171), "../results/results_atcnet/best_weights/best_model_weights.pth"),
        "tresnet_m": (lambda: timm.create_model('tresnet_m', pretrained=True, num_classes=50), "../results/results_tresnet/best_weights/best_model_weights.pth")
    }
    
    if model_name not in model_mapping:
        raise ValueError(f"Model {model_name} not recognized.")
    
    model_class, checkpoint_path = model_mapping[model_name]
    model = model_class()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    # Set the model to evaluation mode
    model.eval()

    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)

    return model


def preprocess_image(image: Image.Image, model_name: str) -> torch.Tensor:
    """
    Preprocesses an image based on the specified model.

    Args:
        image (PIL.Image): Image to preprocess.
        model_name (str): Name of the model that requires preprocessing.

    Returns:
        torch.Tensor: Preprocessed image as a tensor.
    """
    transform_dict = {
        "Glyphnet": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(GLYPHNET_INPUT_SIZE),
            transforms.ToTensor(),
        ]),
        "ATCNet": transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(ATCNET_INPUT_SIZE),
            transforms.ToTensor(),
        ]),
        "tresnet_m": transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(TRESNET_INPUT_SIZE),
            transforms.ToTensor(),
        ])
    }

    if model_name not in transform_dict:
        raise ValueError(f"Model {model_name} not recognized.")

    transform = transform_dict[model_name]
    return transform(image).unsqueeze(0)


def predict(input_tensor: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    """
    Makes a prediction using the provided model.

    Args:
        input_tensor (torch.Tensor): Preprocessed input.
        model (torch.nn.Module): Model to use for prediction.

    Returns:
        np.array: Predicted probabilities for each class.
    """
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        return probabilities.cpu().numpy()


def perturb_image(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Creates perturbed versions of the image based on segments.

    Args:
        image (np.array): Image to perturb.
        segments (np.array): Segments created by the SLIC algorithm.

    Returns:
        np.array: Perturbed images.
    """
    perturbed_images = []
    num_segments = np.max(segments) + 1

    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i
        if np.any(mask):
            mean_value = np.mean(image[mask], axis=0)
            std_value = np.std(image[mask], axis=0)
            perturbed_image[mask] = np.random.normal(loc=mean_value, scale=std_value) if not np.isnan(mean_value).any() and not np.isnan(std_value).any() else mean_value
        perturbed_images.append(perturbed_image)

    return np.array(perturbed_images)


def explain_image_custom(image: Image.Image, model: torch.nn.Module, model_name: str) -> tuple:
    """
    Generates a custom explanation for an image using segment perturbations.

    Args:
        image (PIL.Image): Image to explain.
        model (torch.nn.Module): Model to use for prediction.
        model_name (str): Model name for preprocessing.

    Returns:
        tuple: Image segments and generated explanation.
    """
    image = np.array(image)
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = gray2rgb(image)

    # Segment the image
    segments = slic(image, n_segments=NUM_SEGMENTS, compactness=COMPACTNESS)
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


def explain_image_lime(image: Image.Image, model: torch.nn.Module, model_name: str) -> tuple:
    """
    Generates an explanation for an image using the LIME library.

    Args:
        image (PIL.Image): Image to explain.
        model (torch.nn.Module): Model to use for prediction.
        model_name (str): Model name for preprocessing.

    Returns:
        tuple: Image with explanation overlay and LIME mask.
    """
    def predict_fn(images: np.ndarray) -> np.ndarray:
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


def compare_explanations(image: Image.Image, model: torch.nn.Module, model_name: str) -> None:
    """
    Compares explanations generated by the custom method and the LIME library.

    Args:
        image (PIL.Image): Image to explain.
        model (torch.nn.Module): Model to use for prediction.
        model_name (str): Model name for preprocessing.

    Returns:
        None
    """
    original_image = np.array(image)
    if len(original_image.shape) == 2:
        original_image = gray2rgb(original_image)

    # Custom explanation
    segments_custom, explanation_custom = explain_image_custom(image, model, model_name)

    # LIME explanation
    temp_lime, mask_lime = explain_image_lime(original_image, model, model_name)

    # Create a comparative plot
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


def main_lime(model_name: str) -> None:
    """
    Runs the explanation generation for a set of images using both the custom method and LIME.

    Args:
        model_name (str): Name of the model to use.

    Returns:
        None
    """
    model = load_model(model_name)

    # Define image paths based on the model
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
        logging.info(f"Processing image: {image_path}")
        try:
            # Convert to RGB upon loading
            image = Image.open(image_path).convert('RGB')
            compare_explanations(image, model, model_name)
        except Exception as e:
            logging.error(f"Error processing image {image_path}: {e}")


if __name__ == "__main__":
    for model in ["Glyphnet", "ATCNet", "tresnet_m"]:
        main_lime(model)
