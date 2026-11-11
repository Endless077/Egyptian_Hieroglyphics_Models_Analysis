# Libraries
from utils import *
import logging

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from hydra import main as hydra_main

from codecarbon import EmissionsTracker
from sklearn.model_selection import train_test_split

# Constants
MODEL_WEIGHTS_PATH = "Glyphnet_best_model_weights.pth"
CHECKPOINT_PATH = "checkpoint.bin"
OPTIMIZER_STATE_PATH = "optimizer_state.pth"

# Function to define test transformations
def get_test_transforms():
    return transforms.Compose([
        # Convert to grayscale
        transforms.Grayscale(num_output_channels=1),
        # Convert to tensor
        transforms.ToTensor()
    ])


@hydra_main(config_path="../configs", config_name="config_glyphnet")
def main_Glyphnet(cfg: DictConfig) -> dict:
    """
    Main function for training and evaluating the Glyphnet model.

    Args:
        cfg (DictConfig): Hydra configuration containing data paths, training parameters, etc.

    Returns:
        Dict[str, List[float]]: Metrics collected during training and validation.
    """
    # Start the CodeCarbon tracker
    tracker = EmissionsTracker()
    tracker.start()

    # Load and split data into training and test datasets
    train_labels, test_path, train_path = load_and_split_data(cfg)
    
    logging.info("Splitting data...")
    train_set = GlyphData(root=train_path, class_to_idx=train_labels, transform=train_transform)

    # Split the training set indices into training and validation sets
    train_indices, val_indices, _, _ = train_test_split(
        range(len(train_set)),
        train_set.targets,
        test_size=cfg.data.val_fraction,
        shuffle=True,
        random_state=cfg.model.seed
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"CUDA available? {torch.cuda.is_available()}")

    # Perform grid search for hyperparameter tuning
    best_params, grid_search_results, best_optimizer_state, metrics = perform_grid_search(
        train_set, train_indices, val_indices, param_grid, train_labels, device, cfg, Glyphnet
    )

    if metrics is None:
        logging.error("Metrics were not returned from perform_grid_search, cannot proceed.")
        return

    # Prepare the test set
    test_labels_set = {l for l in [p.strip("/") for p in listdir(test_path)]}
    test_labels = {k: v for k, v in train_labels.items() if k in test_labels_set}

    test_set = GlyphData(
        root=test_path, 
        class_to_idx=test_labels,
        transform=get_test_transforms()
    )

    # Create a DataLoader for the test set
    test_loader = DataLoader(test_set, batch_size=best_params['batch_size'], shuffle=False)

    # Load the best model found during the grid search
    model = Glyphnet(num_classes=len(train_labels))
    try:
        model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    except FileNotFoundError:
        logging.error(f"Model weights file {MODEL_WEIGHTS_PATH} not found.")
        return

    model.to(device)
    
    logging.info("Checking quality on test set:")
    evaluate_model(model, test_loader, device, num_classes=len(train_labels))

    logging.info("Saving the trained model.")
    torch.save(model.state_dict(), CHECKPOINT_PATH)

    logging.info("Saving the optimizer state.")
    torch.save(best_optimizer_state, OPTIMIZER_STATE_PATH)

    # Plot the results of the grid search
    plot_grid_search_results(grid_search_results, param_grid)

    tracker.stop()

    # Plot the learning curves based on the collected metrics
    plot_learning_curves(metrics)

    return metrics


if __name__ == "__main__":
    metrics = main_Glyphnet()
