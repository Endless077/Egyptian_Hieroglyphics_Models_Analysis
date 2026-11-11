# Libraries
import torch
import numpy as np
from lime import lime_image

import timm
import hydra
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
from omegaconf import DictConfig

from skimage.segmentation import mark_boundaries, slic
from skimage.color import gray2rgb
from torchvision import transforms
from PIL import Image

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define transformations for the dataset
data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),

    # Normalization for RGB images
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_ensemble_model(ensemble_model_path: str, device: torch.device, num_classes: int) -> tuple:
    """
    Loads the saved ensemble model and its weights from the specified path.

    Args:
        ensemble_model_path (str): Path to the saved ensemble model checkpoint.
        device (torch.device): The device (CPU or GPU) to load the models onto.
        num_classes (int): Number of output classes for the model.

    Returns:
        tuple: A tuple containing:
            - list: A list of models loaded from the checkpoint.
            - list: Weights associated with each model if available, otherwise None.
    """

    checkpoint = torch.load(ensemble_model_path)
    models = []

    for model_state_dict in checkpoint['models']:
        model = timm.create_model('tresnet_m', pretrained=True, num_classes=num_classes)
        model.load_state_dict(model_state_dict)

        # Set the model to evaluation mode
        model.to(device).eval()
        models.append(model)

    model_weights = checkpoint.get('weights', None)
    return models, model_weights


def predict_ensemble_lime(models: list, image: torch.Tensor, device: torch.device, weights: list = None) -> np.ndarray:
    """
    Performs predictions on an image using the ensemble of models with optional soft voting.

    Args:
        models (list): A list of model instances to use for prediction.
        image (torch.Tensor): The input image tensor to predict on.
        device (torch.device): The device (CPU or GPU) to perform computations on.
        weights (list, optional): A list of weights for each model. If None, unweighted average is used.

    Returns:
        np.ndarray: The average predicted probabilities across the ensemble for each class.
    """

    if len(image.shape) == 3:
        image = image.unsqueeze(0)

    image = image.to(device)
    all_probabilities = []

    for model in models:
        outputs = model(image)

        # Compute probabilities
        probabilities = F.softmax(outputs, dim=1).cpu().detach().numpy()
        all_probabilities.append(probabilities)

    # Combine probabilities using weights if provided
    return np.average(all_probabilities, axis=0, weights=weights) if weights else np.mean(all_probabilities, axis=0)


def perturb_image(image: np.ndarray, segments: np.ndarray) -> np.ndarray:
    """
    Creates perturbed versions of the image based on the segments.

    Args:
        image (np.ndarray): The original image to be perturbed.
        segments (np.ndarray): An array defining segments of the image for perturbation.

    Returns:
        np.ndarray: An array of perturbed images corresponding to each segment.
    """

    perturbed_images = []
    num_segments = np.max(segments) + 1

    for i in range(num_segments):
        perturbed_image = image.copy()
        mask = segments == i
        if np.any(mask):
            mean_value = np.mean(image[mask], axis=0)
            std_value = np.std(image[mask], axis=0)
            perturbed_image[mask] = np.random.normal(loc=mean_value, scale=std_value)
        perturbed_images.append(perturbed_image)

    return np.array(perturbed_images)


def load_and_prepare_image(image_path: str) -> np.ndarray:
    """
    Loads and prepares the image for prediction.

    Args:
        image_path (str): The file path to the image to be loaded.

    Returns:
        np.ndarray: The prepared image in RGB format as a NumPy array.
    """

    logging.info(f"Loading image for interpretation: {image_path}")
    image = Image.open(image_path)
    image = np.array(image)

    # Convert grayscale images to RGB
    if len(image.shape) == 2:
        image = gray2rgb(image)

    return image


def custom_explanation(image_path: str, models: list, model_weights: list, device: torch.device, true_class: int = None) -> int:
    """
    Generates a custom explanation using image segment perturbation.

    Args:
        image_path (str): The file path to the image for which to generate an explanation.
        models (list): A list of model instances for prediction.
        model_weights (list): Weights for the ensemble models.
        device (torch.device): The device to perform computations on.
        true_class (int, optional): The true class of the image, for logging purposes.

    Returns:
        int: The predicted class of the image.
    """

    image = load_and_prepare_image(image_path)
    segments = slic(image, n_segments=150, compactness=10)

    base_image_tensor = data_transforms(Image.fromarray(image)).unsqueeze(0).to(device)

    # Predict using the ensemble of models
    base_prediction = predict_ensemble_lime(models, base_image_tensor, device, model_weights)

    # Extract the class with the highest probability
    top_label = np.argmax(base_prediction[0])
    logging.info(f"Predicted class: {top_label}")
    if true_class is not None:
        logging.info(f"True class: {true_class}")

    perturbed_images = perturb_image(image, segments)

    # Preprocess perturbed images and predict
    perturbed_images_tensor = torch.cat(
        [data_transforms(Image.fromarray(img)).unsqueeze(0) for img in perturbed_images], dim=0).to(device)
    predictions = predict_ensemble_lime(models, perturbed_images_tensor, device, model_weights)

    # Calculate weights based on predictions for the main class
    weights = np.mean(predictions[:, top_label], axis=0)

    # Generate the explanation by mapping the weights to segments
    explanation = np.zeros(segments.shape)
    for i in range(np.max(segments) + 1):
        explanation[segments == i] = weights

    # Display the explanation overlaid on the image
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(image / 255.0, segments))
    plt.imshow(explanation, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.title(f"Custom Explanation for class {top_label}")
    plt.axis('off')
    plt.show()

    return top_label


def lime_library_explanation(image_path: str, models: list, model_weights: list, device: torch.device, true_class: int = None) -> int:
    """
    Generates an explanation using the LIME library.

    Args:
        image_path (str): The file path to the image for which to generate an explanation.
        models (list): A list of model instances for prediction.
        model_weights (list): Weights for the ensemble models.
        device (torch.device): The device to perform computations on.
        true_class (int, optional): The true class of the image, for logging purposes.

    Returns:
        int: The predicted class of the image.
    """

    logging.info(f"Loading image for interpretation: {image_path}")
    image = load_and_prepare_image(image_path)
    image_tensor = data_transforms(Image.fromarray(image))

    def predict_fn(images: list) -> np.ndarray:
        """Prediction function used by LIME."""
        images = [np.uint8(255 * img) if img.dtype == np.float32 else img for img in images]
        images_tensor = torch.stack([data_transforms(Image.fromarray(img)) for img in images], dim=0).to(device)

        return predict_ensemble_lime(models, images_tensor, device, model_weights)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(np.array(image_tensor.permute(1, 2, 0)),
                                             predict_fn,
                                             top_labels=5,
                                             hide_color=0,
                                             num_samples=1000)

    predicted_class = explanation.top_labels[0]
    logging.info(f"Predicted class: {predicted_class}")
    if true_class is not None:
        logging.info(f"True class: {true_class}")

    # Get the explanation and mask
    temp, mask = explanation.get_image_and_mask(predicted_class, positive_only=True, num_features=10,
                                                hide_rest=False)

    # Convert image to RGB if it's grayscale
    if len(temp.shape) == 2:
        temp = gray2rgb(temp)

    # Display the original image with superpixel boundaries
    plt.figure(figsize=(10, 10))
    plt.imshow(mark_boundaries(temp / 255.0, mask))

    # Overlay mask with 'jet' colormap
    plt.imshow(mask, cmap='jet', alpha=0.5)
    plt.title(f'LIME Explanation for class {predicted_class}')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    return predicted_class


def compare_explanations(image_path: str, models: list, model_weights: list, device: torch.device, output_path: str = "comparison_plot.png") -> None:
    """
    Compares the custom explanation with the one generated by the LIME library.

    Args:
        image_path (str): The file path to the image to be explained.
        models (list): A list of model instances for prediction.
        model_weights (list): Weights for the ensemble models.
        device (torch.device): The device to perform computations on.
        output_path (str, optional): The file path to save the comparison plot.

    Returns:
        None: This function saves a comparison plot and displays it.
    """

    # Generate explanations using both methods
    logging.info("Generating custom explanation...")
    custom_predicted_class = custom_explanation(image_path, models, model_weights, device)

    logging.info("Generating LIME library explanation...")
    lime_predicted_class = lime_library_explanation(image_path, models, model_weights, device)

    # Display the results
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    axes[0].imshow(mark_boundaries(custom_explanation(image_path, models, model_weights, device) / 255.0, custom_explanation(image_path, models, model_weights, device)))
    axes[0].set_title(f'Custom Explanation: {custom_predicted_class}')

    axes[1].imshow(mark_boundaries(lime_library_explanation(image_path, models, model_weights, device) / 255.0, lime_library_explanation(image_path, models, model_weights, device)))
    axes[1].set_title(f'LIME Explanation: {lime_predicted_class}')

    plt.suptitle("Comparison of Explanations")
    plt.tight_layout()
    plt.savefig(output_path)
    logging.info(f"Comparison plot saved at {output_path}")
    plt.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function to run the ensemble model explanation.

    Args:
        cfg (DictConfig): Configuration object containing model and data paths, and output settings.

    Returns:
        None: This function runs the explanation process and visualizes the results.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, model_weights = load_ensemble_model(cfg.model.ensemble_path, device, cfg.model.num_classes)
    image_path = cfg.data.image_path

    # Compare custom and LIME explanations
    compare_explanations(image_path, models, model_weights, device, cfg.output.plot_path)


if __name__ == "__main__":
    main()
