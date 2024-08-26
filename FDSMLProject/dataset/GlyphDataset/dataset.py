# coding: utf-8

from typing import Dict, Tuple, List
from torchvision.datasets import ImageFolder


class GlyphData(ImageFolder):
    def __init__(self, class_to_idx: Dict[str, int], root: str = "../datasets/prepared_data/train/", *args, **kwargs):
        """
        Inizializza il dataset GlyphData come un'estensione di ImageFolder di torchvision,
        utilizzando una mappatura personalizzata delle classi.

        Args:
            class_to_idx (Dict[str, int]): Mappatura personalizzata delle etichette (stringhe) agli ID (interi).
            root (str): Directory contenente i dati di addestramento o test. Default è "prepared_data/train/".
            *args: Argomenti addizionali per la classe base ImageFolder.
            **kwargs: Argomenti addizionali per la classe base ImageFolder.
        """
        # Inizializza una lista di classi con la dimensione basata sull'ID massimo
        # Ogni elemento della lista è inizialmente "UNKNOWN"
        self.classes_list = ["UNKNOWN" for _ in range(max(class_to_idx.values()) + 1)]

        # Salva la mappatura personalizzata delle classi
        self.classes_map = class_to_idx

        # Popola la lista delle classi sostituendo "UNKNOWN" con i nomi delle classi corrette
        for k, v in class_to_idx.items():
            self.classes_list[v] = k

        # Inizializza la classe base ImageFolder con i parametri passati
        super(GlyphData, self).__init__(root=root, *args, **kwargs)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Sovrascrive il metodo find_classes di ImageFolder per restituire le classi e la mappatura personalizzata.

        Args:
            directory (str): Percorso alla directory contenente le immagini organizzate per classi.

        Returns:
            Tuple[List[str], Dict[str, int]]: Una tupla contenente la lista delle classi (in ordine di ID)
                                              e il dizionario di mappatura delle classi.
        """
        # Restituisce la lista delle classi e la mappatura personalizzata delle classi
        return self.classes_list, self.classes_map
