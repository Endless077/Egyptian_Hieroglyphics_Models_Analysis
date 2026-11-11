# Libraries
# coding: utf-8
from typing import Dict, Tuple, List
from torchvision.datasets import ImageFolder


class GlyphData(ImageFolder):
    def __init__(self, class_to_idx: Dict[str, int], root: str = "../datasets/prepared_data/train/", *args, **kwargs):
        """
        Initializes the GlyphData dataset as an extension of ImageFolder from torchvision,
        using a custom class mapping.

        Args:
            class_to_idx (Dict[str, int]): Custom mapping of labels (strings) to IDs (integers).
            root (str): Directory containing training or test data. Default is "../datasets/prepared_data/train/".
            *args: Additional arguments for the base ImageFolder class.
            **kwargs: Additional keyword arguments for the base ImageFolder class.
        """
        # Initialize classes_list based on the maximum ID found in class_to_idx
        self.classes_list = [""] * (max(class_to_idx.values()) + 1)

        # Save the custom class mapping
        self.classes_map = class_to_idx

        # Populate the class list with actual class names
        for class_name, class_id in class_to_idx.items():
            self.classes_list[class_id] = class_name

        # Initialize the base ImageFolder class with the provided parameters
        super().__init__(root=root, *args, **kwargs)


    def find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """
        Overrides the find_classes method of ImageFolder to return the classes and custom mapping.

        Args:
            directory (str): Path to the directory containing images organized by class.

        Returns:
            Tuple[List[str], Dict[str, int]]: A tuple containing the list of classes (in order of ID)
                                              and the class mapping dictionary.
        """
        return self.classes_list, self.classes_map
