import hashlib
import shutil
import logging
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image

UNKNOWN_LABEL = "UNKNOWN"

def augment_image(image_path, save_to_dir, augmentor, augment_count=5):
    """
    Applica tecniche di data augmentation a un'immagine e salva le immagini augmentate.

    Args:
        image_path (str): Il percorso dell'immagine originale.
        save_to_dir (str): La directory in cui salvare le immagini augmentate.
        augmentor (torchvision.transforms.Compose): Le trasformazioni da applicare per la data augmentation.
        augment_count (int): Il numero di immagini augmentate da generare per ciascuna immagine originale.
    """
    image = Image.open(image_path)
    for i in range(augment_count):
        augmented_image = augmentor(image)
        augmented_image.save(join(save_to_dir, f'aug_{i}_{hashlib.md5(image_path.encode("utf-8")).hexdigest()}.png'))

if __name__ == "__main__":

    from argparse import ArgumentParser

    # Parsing degli argomenti da linea di comando
    ap = ArgumentParser()
    ap.add_argument("--data_path", default="/".join(("data", "Dataset", "Manual", "Preprocessed")),
                    help="Percorso alla directory contenente le immagini preprocessate.")
    ap.add_argument("--prepared_data_path", default="prepared_data",
                    help="Percorso dove salvare il dataset preparato.")
    ap.add_argument("--balanced_data_path", default="balanced_data",
                    help="Percorso dove salvare il dataset bilanciato con data augmentation.")
    ap.add_argument("--test_fraction", type=float, default=0.2,
                    help="Frazione del dataset da utilizzare come set di test.")
    ap.add_argument("--seed", type=int, default=261,
                    help="Seed per la riproducibilità del random split.")

    arguments = ap.parse_args()

    # Impostazione del logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Preparazione dei percorsi
    file_dir = dirname(__file__)
    stele_path = join(file_dir, arguments.data_path)
    steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]

    res_image_paths, labels = [], []

    # Raccolta di tutti i percorsi delle immagini e delle relative etichette
    for stele in steles:
        image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]
        for path in image_paths:
            res_image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

    list_of_paths = np.asarray(res_image_paths)
    labels = np.array(labels)

    logging.debug(f"Numero totale di etichette uniche: {len(set(labels))}")

    # Identificazione delle etichette che compaiono solo una volta
    labels_just_once = np.array([l for (l, c) in Counter(labels).items() if c <= 1])
    logging.debug(f"Numero di etichette viste solo una volta: {len(labels_just_once)}")

    # Le immagini con etichette viste solo una volta saranno aggiunte al set di addestramento
    to_be_added_to_train_only = np.nonzero(np.isin(labels, labels_just_once))[0]

    # Le immagini con etichette sconosciute saranno rimosse
    to_be_deleted = np.nonzero(labels == UNKNOWN_LABEL)[0]

    # Rimuove gli elementi con etichette sconosciute o con etichette viste solo una volta
    to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only])
    filtered_list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
    filtered_labels = np.delete(labels, to_be_deleted, 0)

    # Suddivisione dei dati in set di addestramento e test
    train_paths, test_paths, y_train, y_test = train_test_split(
        filtered_list_of_paths,
        filtered_labels,
        stratify=filtered_labels,
        test_size=arguments.test_fraction,
        random_state=arguments.seed
    )

    # Aggiunta delle etichette a occorrenza singola al set di addestramento
    train_paths = np.concatenate([train_paths, list_of_paths[to_be_added_to_train_only]])
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

    # Creazione delle cartelle per i dati bilanciati
    makedirs(arguments.balanced_data_path, exist_ok=True)
    train_balanced_dir = join(arguments.balanced_data_path, "train")
    test_balanced_dir = join(arguments.balanced_data_path, "test")
    [makedirs(join(train_balanced_dir, l), exist_ok=True) for l in set(y_train)]
    [makedirs(join(test_balanced_dir, l), exist_ok=True) for l in set(y_test)]

    # Configurazione del data generator per la data augmentation
    augmentor = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05)),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
    ])

    # Copia e applicazione della data augmentation alle immagini del set di addestramento
    for fp, label in zip(train_paths, y_train):
        target_dir = join(train_balanced_dir, label)
        fn = join(target_dir, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)
        # Applica data augmentation
        augment_image(fn, target_dir, augmentor)

    # Copia delle immagini nel set di test (senza data augmentation)
    for fp, label in zip(test_paths, y_test):
        target_dir = join(test_balanced_dir, label)
        fn = join(target_dir, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)

    logging.info("Dataset suddiviso e data augmentation applicata con successo nella cartella 'balanced_data'.")
