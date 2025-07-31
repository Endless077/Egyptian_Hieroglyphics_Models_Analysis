# Libraries
import logging
import subprocess
from codecarbon import EmissionsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)


def train_yolov5(data_yaml: str, epochs: int = 50, batch_size: int = 16, img_size: int = 640, weights: str = "yolov5s.pt") -> None:
    """
    Function to train a YOLOv5 model using a training script.

    Args:
        data_yaml (str): Path to the YAML file containing the data configuration for training.
        epochs (int): Number of epochs to train the model (default: 50).
        batch_size (int): The batch size used during training (default: 16).
        img_size (int): The size of the input images (default: 640).
        weights (str): The path to the model weights (default: "yolov5s.pt").
    """
    # Validate inputs
    if epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")
    if batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if img_size <= 0:
        raise ValueError("Image size must be a positive integer.")

    # Build the command for training YOLOv5
    command = [
        "python", "../yolov5/train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", data_yaml,
        "--weights", weights,
        "--cache"
    ]

    # Execute the command in the terminal
    try:
        logging.info("Starting YOLOv5 training...")
        subprocess.run(command, check=True)
        logging.info("Training completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error occurred during training: {e}")


if __name__ == "__main__":
    # Start monitoring CO2 emissions with CodeCarbon
    tracker = EmissionsTracker()
    tracker.start()

    # Train the YOLOv5 model
    train_yolov5(data_yaml='../configs/data.yaml', epochs=100, batch_size=4, img_size=320)

    # Stop monitoring CO2 emissions
    tracker.stop()
