# Libraries
import logging
import hashlib
import shutil

from collections import Counter
from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
from torchvision import transforms
from PIL import Image
from typing import List, Tuple, Namespace

# Constants
UNKNOWN_LABEL = "UNKNOWN"
IMAGE_EXTENSION = ".png"


def augment_image(image_path: str, save_to_dir: Path, augmentor: transforms.Compose, augment_count: int = 5) -> None:
    """
    Applies data augmentation techniques to an image and saves the augmented images.

    Args:
        image_path (str): The path of the original image.
        save_to_dir (Path): The directory where augmented images will be saved.
        augmentor (torchvision.transforms.Compose): Transformations to apply for data augmentation.
        augment_count (int): The number of augmented images to generate for each original image.
    """
    image = Image.open(image_path)
    for i in range(augment_count):
        augmented_image = augmentor(image)
        augmented_image.save(save_to_dir / f'aug_{i}_{hashlib.md5(image_path.encode("utf-8")).hexdigest()}{IMAGE_EXTENSION}')


def collect_image_paths(stele_path: Path) -> Tuple[List[Path], List[str]]:
    """
    Collects all image paths and their corresponding labels from the given directory.

    Args:
        stele_path (Path): Path to the directory containing images.

    Returns:
        Tuple[List[Path], List[str]]: Image paths and corresponding labels.
    """
    image_paths, labels = [], []
    for stele in stele_path.iterdir():
        if stele.is_dir():
            for img_file in stele.glob('*'):
                if img_file.is_file():
                    image_paths.append(img_file)
                    # Extract label from filename
                    labels.append(img_file.stem.split('_')[-1])
    return image_paths, labels


def setup_logging() -> None:
    """
    Configures the logging settings for the application.

    This function sets the logging level to DEBUG and specifies the format for the log messages.
    The format includes the timestamp, log level, and the message itself.

    Returns:
        None
    """
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def create_directory(path: str) -> None:
    """
    Creates a directory if it doesn't exist.

    Args:
        path (str): The directory path to create.

    Returns:
        None
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def main(arguments: Namespace) -> None:
    """
    Main function to prepare the dataset by collecting images, filtering labels, splitting the dataset,
    and applying data augmentation.

    Args:
        arguments (Namespace): The command line arguments containing paths and options.

    Returns:
        None
    """
    setup_logging()

    # Prepare paths
    stele_path = Path(arguments.data_path)
    image_paths, labels = collect_image_paths(stele_path)

    labels = np.array(labels)
    logging.debug(f"Total number of unique labels: {len(set(labels))}")

    # Identify and filter labels
    labels_counter = Counter(labels)
    labels_just_once = [label for label, count in labels_counter.items() if count <= 1]
    logging.debug(f"Number of labels seen only once: {len(labels_just_once)}")

    # Filter paths based on labels
    to_be_added_to_train_only = [i for i, label in enumerate(labels) if label in labels_just_once]
    to_be_deleted = [i for i, label in enumerate(labels) if label == UNKNOWN_LABEL]
    to_be_deleted.extend(to_be_added_to_train_only)

    filtered_image_paths = np.delete(image_paths, to_be_deleted, axis=0)
    filtered_labels = np.delete(labels, to_be_deleted, axis=0)

    # Split data into training and test sets
    train_paths, test_paths, y_train, y_test = train_test_split(
        filtered_image_paths,
        filtered_labels,
        stratify=filtered_labels,
        test_size=arguments.test_fraction,
        random_state=arguments.seed
    )

    # Adding single-occurrence labels to the training set
    train_paths = np.concatenate([train_paths, filtered_image_paths[to_be_added_to_train_only]])
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

    # Create directories for balanced data
    create_directory(arguments.balanced_data_path)
    train_balanced_dir = Path(arguments.balanced_data_path) / "train"
    test_balanced_dir = Path(arguments.balanced_data_path) / "test"
    create_directory(train_balanced_dir)
    create_directory(test_balanced_dir)

    for label in set(y_train):
        create_directory(train_balanced_dir / label)
    for label in set(y_test):
        create_directory(test_balanced_dir / label)

    # Setting up the data generator for data augmentation
    augmentor = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ])

    # Copy and apply data augmentation to training set images
    for fp, label in zip(train_paths, y_train):
        target_dir = train_balanced_dir / label
        fn = target_dir / f"{hashlib.md5(fp.encode('utf-8')).hexdigest()}{IMAGE_EXTENSION}"
        
        try:
            shutil.copyfile(fp, fn)
            augment_image(fn, target_dir, augmentor)
        except Exception as e:
            logging.error(f"Error processing {fp}: {e}")

    # Copy test set images without data augmentation
    for fp, label in zip(test_paths, y_test):
        target_dir = test_balanced_dir / label
        fn = target_dir / f"{hashlib.md5(fp.encode('utf-8')).hexdigest()}{IMAGE_EXTENSION}"
        shutil.copyfile(fp, fn)

    logging.info("Dataset split and data augmentation successfully applied in the 'balanced_data' folder.")


if __name__ == "__main__":
    # Parsing command-line arguments
    parser = ArgumentParser()
    parser.add_argument("--data_path", default="../datasets/data/Dataset/Manual/Preprocessed", 
                        help="Path to the directory containing preprocessed images.")
    parser.add_argument("--prepared_data_path", default="../dataset/prepared_data",
                        help="Path to save the prepared dataset.")
    parser.add_argument("--balanced_data_path", default="../datasets/balanced_data",
                        help="Path to save the balanced dataset with data augmentation.")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Fraction of the dataset to use as the test set.")
    parser.add_argument("--seed", type=int, default=261,
                        help="Seed for reproducibility of the random split.")

    args = parser.parse_args()
    main(args)
