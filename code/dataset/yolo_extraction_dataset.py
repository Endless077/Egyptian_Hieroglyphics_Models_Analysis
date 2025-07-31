# Libraries
import cv2
from pathlib import Path

# Constants
DATASET_SPLITS = ['train', 'valid', 'test']
IMAGE_EXTENSION = '.jpg'
LABEL_EXTENSION = '.txt'

# Class mapping
class_to_idx = {
    # Mapping of classes to their respective numeric labels
    'A55': 0, 'Aa15': 1, 'Aa26': 2, 'Aa27': 3, 'Aa28': 4, 'D1': 5, 'D10': 6, 'D156': 7, 'D19': 8,
    'D2': 9, 'D21': 10, 'D28': 11, 'D34': 12, 'D35': 13, 'D36': 14, 'D39': 15, 'D4': 16, 'D46': 17,
    'D52': 18, 'D53': 19, 'D54': 20, 'D56': 21, 'D58': 22, 'D60': 23, 'D62': 24, 'E1': 25, 'E17': 26,
    'E23': 27, 'E34': 28, 'E9': 29, 'F12': 30, 'F13': 31, 'F16': 32, 'F18': 33, 'F21': 34, 'F22': 35,
    'F23': 36, 'F26': 37, 'F29': 38, 'F30': 39, 'F31': 40, 'F32': 41, 'F34': 42, 'F35': 43, 'F4': 44,
    'F40': 45, 'F9': 46, 'G1': 47, 'G10': 48, 'G14': 49, 'G17': 50, 'G21': 51, 'G25': 52, 'G26': 53,
    'G29': 54, 'G35': 55, 'G36': 56, 'G37': 57, 'G39': 58, 'G4': 59, 'G40': 60, 'G43': 61, 'G5': 62,
    'G50': 63, 'G7': 64, 'H6': 65, 'I10': 66, 'I5': 67, 'I9': 68, 'L1': 69, 'M1': 70, 'M12': 71,
    'M16': 72, 'M17': 73, 'M18': 74, 'M195': 75, 'M20': 76, 'M23': 77, 'M26': 78, 'M29': 79, 'M3': 80,
    'M4': 81, 'M40': 82, 'M41': 83, 'M42': 84, 'M44': 85, 'M8': 86, 'N1': 87, 'N14': 88, 'N16': 89,
    'N17': 90, 'N18': 91, 'N19': 92, 'N2': 93, 'N24': 94, 'N25': 95, 'N26': 96, 'N29': 97, 'N30': 98,
    'N31': 99, 'N35': 100, 'N36': 101, 'N37': 102, 'N41': 103, 'N5': 104, 'O1': 105, 'O11': 106,
    'O28': 107, 'O29': 108, 'O31': 109, 'O34': 110, 'O4': 111, 'O49': 112, 'O50': 113, 'O51': 114,
    'P1': 115, 'P13': 116, 'P6': 117, 'P8': 118, 'P98': 119, 'Q1': 120, 'Q3': 121, 'Q7': 122, 'R4': 123,
    'R8': 124, 'S24': 125, 'S28': 126, 'S29': 127, 'S34': 128, 'S42': 129, 'T14': 130, 'T20': 131,
    'T21': 132, 'T22': 133, 'T28': 134, 'T30': 135, 'U1': 136, 'U15': 137, 'U28': 138, 'U33': 139,
    'U35': 140, 'U7': 141, 'V13': 142, 'V16': 143, 'V22': 144, 'V24': 145, 'V25': 146, 'V28': 147,
    'V30': 148, 'V31': 149, 'V4': 150, 'V6': 151, 'V7': 152, 'W11': 153, 'W14': 154, 'W15': 155,
    'W18': 156, 'W19': 157, 'W22': 158, 'W24': 159, 'W25': 160, 'X1': 161, 'X6': 162, 'X8': 163,
    'Y1': 164, 'Y2': 165, 'Y3': 166, 'Y5': 167, 'Z1': 168, 'Z11': 169, 'Z7': 170
}


def convert_yolo_to_classification(dataset_dir: str = '../datasets/dataset_yolo', output_dir: str = '../datasets/classification_dataset') -> None:
    """
    Converts a YOLOv5 annotated dataset into a classification dataset by cropping images
    of bounding boxes and saving them into folders based on their class.

    Args:
        dataset_dir (str): Path to the original YOLOv5 dataset directory.
        output_dir (str): Path to the output directory where the classification dataset will be saved.

    Returns:
        None
    """
    def normalize_to_pixel_coords(coords: list[float], img_width: int, img_height: int) -> tuple[int, int, int, int]:
        """
        Converts normalized coordinates (YOLO format) to pixel coordinates.

        Args:
            coords (list[float]): List of normalized coordinates (x_center, y_center, width, height).
            img_width (int): Width of the image.
            img_height (int): Height of the image.

        Returns:
            tuple[int, int, int, int]: A tuple containing pixel coordinates (x_min, y_min, x_max, y_max).
        """
        x_center, y_center, width, height = coords
        x_min = int((x_center - width / 2) * img_width)
        y_min = int((y_center - height / 2) * img_height)
        x_max = int((x_center + width / 2) * img_width)
        y_max = int((y_center + height / 2) * img_height)
        return x_min, y_min, x_max, y_max

    # Process each part of the dataset (train, val, test)
    for split in DATASET_SPLITS:
        split_images_dir = Path(dataset_dir) / split / 'images'
        split_labels_dir = Path(dataset_dir) / split / 'labels'
        split_output_dir = Path(output_dir) / split

        # Create the output structure
        split_output_dir.mkdir(parents=True, exist_ok=True)

        # Process each image and annotation file
        for label_file in split_labels_dir.glob('*' + LABEL_EXTENSION):
            image_path = split_images_dir / (label_file.stem + IMAGE_EXTENSION)

            # Check if the image exists before processing
            if not image_path.is_file():
                print(f"Warning: Image {image_path} does not exist.")
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Error: Could not read image {image_path}. Skipping.")
                continue
            
            img_height, img_width = image.shape[:2]

            with open(label_file, 'r') as f:
                annotations = f.readlines()

            for annotation in annotations:
                values = list(map(float, annotation.split()))
                class_id = int(values[0])
                
                # All coordinate pairs
                coords = values[1:]

                # Check for valid number of coordinates
                if len(coords) < 4:
                    print(f"Error: Insufficient coordinate values in {annotation.strip()}. Skipping.")
                    continue
                
                # Normalize coordinates to pixel coordinates
                x_min, y_min, x_max, y_max = normalize_to_pixel_coords(coords, img_width, img_height)

                # Validate bounding box coordinates
                if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
                    print(f"Warning: Bounding box coordinates are out of image bounds for {annotation.strip()}. Skipping.")
                    continue

                # Crop the image using the bounding box
                cropped_img = image[y_min:y_max, x_min:x_max]

                # Ensure the cropped image has valid dimensions
                if cropped_img.size == 0:
                    print(f"Warning: Cropped image for {annotation.strip()} is empty. Skipping.")
                    continue

                # Save the cropped image in the corresponding class folder
                class_folder = split_output_dir / str(class_id)
                class_folder.mkdir(parents=True, exist_ok=True)

                # Generate a unique name for the cropped image
                img_name = f"{label_file.stem}_{x_min}_{y_min}.jpg"
                cv2.imwrite(str(class_folder / img_name), cropped_img)

    print("Conversion completed!")
