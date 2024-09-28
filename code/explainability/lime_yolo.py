# Libraries
import warnings
import cv2
import torch
import numpy as np

from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
import logging

# Disable warnings to keep the output clean
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
# (update based on your model's configuration)
IMAGE_SIZE = (640, 640)
NUM_CLASSES = 80

# Random colors for classes (used for drawing bounding boxes)
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))


def parse_detections(results) -> tuple:
    """
    Extracts bounding boxes, colors, and class names from YOLOv5 model results.

    Args:
        results (object): YOLOv5 model inference results.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], List[str]]:
        - boxes: Bounding box coordinates.
        - colors: Colors associated with each class.
        - names: Names of detected classes.
    """
    detections = results.pandas().xyxy[0].to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            # Ignore predictions with low confidence
            continue
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes: list, colors: list, names: list, img: np.ndarray) -> np.ndarray:
    """
    Draws bounding boxes and class names on the image.

    Args:
        boxes (List[Tuple[int, int, int, int]]): Bounding box coordinates.
        colors (List[np.ndarray]): Colors associated with each class.
        names (List[str]): Names of detected classes.
        img (np.ndarray): Image on which to draw the bounding boxes.

    Returns:
        np.ndarray: Image with drawn bounding boxes and class names.
    """
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(img, name, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, lineType=cv2.LINE_AA)
    return img


def renormalize_cam_in_bounding_boxes(boxes: list[tuple[int, int, int, int]], 
                                      colors: list[np.ndarray], 
                                      names: list[str], 
                                      image_float_np: np.ndarray, 
                                      grayscale_cam: np.ndarray) -> np.ndarray:
    """
    Renormalizes the CAM within the bounding boxes and draws bounding boxes on the image.

    Args:
        boxes (List[Tuple[int, int, int, int]]): Bounding box coordinates.
        colors (List[np.ndarray]): Colors associated with each class.
        names (List[str]): Names of detected classes.
        image_float_np (np.ndarray): Normalized image in float format.
        grayscale_cam (np.ndarray): Grayscale CAM.

    Returns:
        np.ndarray: Image with renormalized CAM and drawn bounding boxes.
    """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    return draw_detections(boxes, colors, names, eigencam_image_renormalized)


def load_image(image_path: str) -> np.ndarray:
    """
    Loads an image from a specified path and handles errors.

    Args:
        image_path (str): The file path to the image to be loaded.

    Returns:
        np.ndarray: The loaded image as a NumPy array.

    Raises:
        UnidentifiedImageError: If the image cannot be identified or loaded.
    """
    try:
        img = np.array(Image.open(image_path))
        return img
    except UnidentifiedImageError as e:
        logging.error(f"Failed to load image from path: {image_path}. Error: {e}")
        raise


def preprocess_image(img: np.ndarray) -> torch.Tensor:
    """
    Prepares the image for inference by resizing and normalizing it.

    Args:
        img (np.ndarray): The input image as a NumPy array.

    Returns:
        torch.Tensor: The preprocessed image as a PyTorch tensor with an added batch dimension.
    """
    img = cv2.resize(img, IMAGE_SIZE)

    # Normalize the image
    img = np.float32(img) / 255
    transform = transforms.ToTensor()

    # Convert to tensor and add a batch dimension
    return transform(img).unsqueeze(0)


def main(image_path: str, model_path: str) -> None:
    """
    Main function to run the image analysis and Class Activation Map (CAM) generation.

    Args:
        image_path (str): The file path to the image for analysis.
        model_path (str): The file path to the trained YOLOv5 model.

    Returns:
        None: This function performs inference, generates CAMs, and displays/saves the resulting images.
    """
    img = load_image(image_path)
    tensor = preprocess_image(img)

    # Load the trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    # Set the model to evaluation mode and move to CPU
    model.eval().cpu()
    # Specify target layers for CAM
    target_layers = [model.model.model.model[-2]]

    # Perform inference on the custom image
    results = model([img])
    boxes, colors, names = parse_detections(results)
    
    # Generate the CAM (Class Activation Map) using EigenCAM
    cam = EigenCAM(model, target_layers)
    grayscale_cam = cam(tensor)[0, :, :]  # Get the CAM as a grayscale image
    cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

    # Save and display the CAM image
    cam_image_pil = Image.fromarray(cam_image)
    cam_image_pil.save("cam_image_custom.png")
    cam_image_pil.show()

    # Renormalize the CAM within the bounding boxes
    renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)
    renormalized_cam_image_pil = Image.fromarray(renormalized_cam_image)
    renormalized_cam_image_pil.save("renormalized_cam_image_custom.png")
    renormalized_cam_image_pil.show()

    # Combine and display all images side by side
    combined_image = np.hstack((img, cam_image, renormalized_cam_image))
    combined_image_pil = Image.fromarray(combined_image)
    combined_image_pil.save("combined_image_custom.png")
    combined_image_pil.show()


# Specify paths for the image and model
image_path = "../datasets/dataset_yolo/train/images/Screen-Shot-2020-07-06-at-4-08-53-PM_0_png.rf.93d6504cc1c64701273f399044eecf4d.jpg"
model_path = "../results/results_yolov5/best_weights/best.pt"


if __name__ == "__main__":
    main(image_path, model_path)
