# Libraries
from utils import *
import logging

import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from omegaconf import DictConfig
from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split


@hydra.main(config_path="../configs", config_name="config_atcnet")
def main_ATCNet(cfg: DictConfig):
    """
    Main function for training and evaluating the ATCNet model.

    Args:
        cfg (DictConfig): Hydra configuration containing data paths, training parameters, etc.

    Returns:
        Dict[str, List[float]]: Metrics collected during training and validation.
    """
    # Start the CodeCarbon tracker to monitor carbon emissions during the training process
    # (optionally, specify an output directory if needed)
    tracker = EmissionsTracker()
    tracker.start()

    # Load and split data into training and test datasets
    train_labels, test_path, train_path = load_and_split_data(cfg)

    # Create a dataset object for the training set using ImageFolder
    train_set = datasets.ImageFolder(root=train_path, transform=train_transform)

    # Log the data loading and splitting process
    logging.info("Data loaded and splitting into training and validation sets...")

    # Split the training set indices into training and validation sets
    train_indices, val_indices = train_test_split(
        # Create a range of indices for the dataset
        range(len(train_set)),
        # Fraction of data to reserve for validation
        test_size=cfg.data.val_fraction,
        # Set a random seed for reproducibility
        random_state=cfg.model.seed
    )

    # Log whether CUDA (GPU) is available for training
    logging.info(f"CUDA available? {torch.cuda.is_available()}")

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Perform grid search for hyperparameter tuning using the ATCNet model class
    best_params, grid_search_results, best_optimizer_state, metrics = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, ATCNet
    )

    # Check if metrics were returned; if not, log an error and exit
    if metrics is None:
        logging.error("Metrics were not returned from perform_grid_search, cannot proceed.")
        return {}

    # Create a dataset object for the test set using ImageFolder
    test_set = datasets.ImageFolder(root=test_path, transform=train_transform)

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    # Initialize the ATCNet model with the number of classes
    model = ATCNet(n_classes=len(train_set.classes))

    # Load the best model weights found during the training process
    try:
        model.load_state_dict(torch.load("ATCNet_best_model_weights.pth"))
        logging.info("Best model weights loaded successfully.")
    except FileNotFoundError:
        logging.error("Model weights file not found. Please ensure it exists.")
        return {}

    model.to(device)  # Move model to the appropriate device (CPU or GPU)

    # Log the evaluation process
    logging.info("Evaluating model on test set...")

    # Evaluate the model
    evaluate_model(model, test_loader, device, num_classes=len(train_set.classes))

    # Log the model saving process
    logging.info("Saving the trained model.")

    # Save the model weights to a file
    torch.save(model.state_dict(), "ATCNet_checkpoint.bin")

    # Plot the results of the grid search
    plot_grid_search_results(grid_search_results, param_grid)

    # Stop the CodeCarbon tracker
    tracker.stop()

    # Plot the learning curves based on the collected metrics
    plot_learning_curves(metrics)

    return metrics


if __name__ == "__main__":
    metrics = main_ATCNet()
