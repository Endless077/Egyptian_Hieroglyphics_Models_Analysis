# Libraries
# coding: utf-8
import logging
import hashlib
import shutil
import numpy as np

from collections import Counter
from os import listdir, makedirs
from argparse import ArgumentParser
from os.path import isdir, isfile, join, dirname
from sklearn.model_selection import train_test_split

# Label for unknown data
UNKNOWN_LABEL = "UNKNOWN"


def setup_logging() -> None:
    """
    Set up logging configuration.

    Configures the logging settings for the application to display debug and higher level messages
    along with a timestamp and the severity level of the log messages.
    
    Returns:
        None
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def get_image_paths_and_labels(stele_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Get image paths and labels from the given stele directory.

    Args:
        stele_path (str): The directory path containing subdirectories of images.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of image file paths.
            - np.ndarray: An array of labels extracted from the image file names.
    """
    res_image_paths = []
    labels = []
    
    steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]
    
    for stele in steles:
        image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]
        
        for path in image_paths:
            res_image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1): path.rfind(".")])
    
    return np.array(res_image_paths), np.array(labels)


def filter_labels(labels: np.ndarray) -> tuple[np.ndarray, set[str]]:
    """
    Filter labels and identify unknown or single occurrence labels.

    Args:
        labels (np.ndarray): An array of labels extracted from image file names.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Labels that appear only once in the dataset.
            - set: A set of labels that appear only once.
    """
    label_counts = Counter(labels)
    labels_just_once = {label for label, count in label_counts.items() if count <= 1}
    logging.debug(f"Labels that appear only once: {len(labels_just_once)}")
    
    return np.array([l for l in labels if l in labels_just_once]), labels_just_once


def main() -> None:
    """
    Main function to prepare the dataset for training and testing.

    Parses command line arguments, sets up logging, retrieves image paths and labels,
    filters the labels, splits the dataset into training and test sets, and organizes the
    images into directories for each label.

    Returns:
        None
    """
    ap = ArgumentParser()
    ap.add_argument("--data_path", default=join("..", "datasets", "data", "Dataset", "Manual", "Preprocessed"),
                    help="Path to the directory of pre-processed data")
    ap.add_argument("--prepared_data_path", default="../datasets/prepared_data",
                    help="Path to the directory to save the prepared data")
    ap.add_argument("--test_fraction", type=float, default=0.2,
                    help="Fraction of the dataset to be used as the test set")
    ap.add_argument("--seed", type=int, default=261,
                    help="Seed for randomization during dataset splitting")

    arguments = ap.parse_args()
    setup_logging()

    # Path to the current file directory
    file_dir = dirname(__file__)
    stele_path = join(file_dir, arguments.data_path)

    # Get image paths and labels
    res_image_paths, labels = get_image_paths_and_labels(stele_path)
    logging.debug(f"Total number of labels: {len(set(labels))}")

    # Identify unknown labels and those appearing only once
    to_be_added_to_train_only, labels_just_once = filter_labels(labels)
    to_be_deleted = np.nonzero(labels == UNKNOWN_LABEL)[0]
    to_be_added_to_train_only_indices = np.nonzero(np.isin(labels, labels_just_once))[0]

    # Remove elements to be added only to the training set or that are unknown
    to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only_indices])
    filtered_list_of_paths = np.delete(res_image_paths, to_be_deleted, 0)
    filtered_labels = np.delete(labels, to_be_deleted, 0)

    # Split the dataset into training and test sets
    train_paths, test_paths, y_train, y_test = train_test_split(
        filtered_list_of_paths,
        filtered_labels,
        stratify=filtered_labels,
        test_size=arguments.test_fraction,
        random_state=arguments.seed
    )

    # Add to the training set the labels that appear only once
    train_paths = np.concatenate([train_paths, res_image_paths[to_be_added_to_train_only_indices]])
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only_indices]])

    # Create directories for prepared data
    makedirs(arguments.prepared_data_path, exist_ok=True)
    for l in set(y_train):
        makedirs(join(arguments.prepared_data_path, "train", l), exist_ok=True)
    for l in set(y_test):
        makedirs(join(arguments.prepared_data_path, "test", l), exist_ok=True)

    # Copy images to the respective training and test sets
    for fp, label in zip(train_paths, y_train):
        fn = join(arguments.prepared_data_path, "train", label, f"{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")
        shutil.copyfile(fp, fn)

    for fp, label in zip(test_paths, y_test):
        fn = join(arguments.prepared_data_path, "test", label, f"{hashlib.md5(fp.encode('utf-8')).hexdigest()}.png")
        shutil.copyfile(fp, fn)


if __name__ == "__main__":
    main()
