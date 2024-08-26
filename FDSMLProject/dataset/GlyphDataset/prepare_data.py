# coding: utf-8
"""
Costruzione di un dataset seguendo l'approccio del repository `glyphreader`.
Il dataset viene costruito e diviso utilizzando `train_test_split` da `sklearn`,
quindi la divisione esatta come in `glyphreader` potrebbe non essere riproducibile.
Pertanto, facciamo la nostra divisione.
"""

import hashlib
import shutil
import logging
from collections import Counter
from os import listdir, makedirs
from os.path import isdir, isfile, join, dirname

import numpy as np
from sklearn.model_selection import train_test_split

UNKNOWN_LABEL = "UNKNOWN"  # Etichetta per i dati sconosciuti

if __name__ == "__main__":

    from argparse import ArgumentParser

    # Parser degli argomenti da riga di comando
    ap = ArgumentParser()
    ap.add_argument("--data_path", default="../datasets".join(("data", "Dataset", "Manual", "Preprocessed")),
                    help="Percorso alla directory dei dati pre-processati")
    ap.add_argument("--prepared_data_path", default="../datasets/prepared_data",
                    help="Percorso alla directory per salvare i dati preparati")
    ap.add_argument("--test_fraction", type=float, default=0.2,
                    help="Frazione del dataset da usare come set di test")
    ap.add_argument("--seed", type=int, default=261,
                    help="Seed per la randomizzazione nella suddivisione del dataset")

    arguments = ap.parse_args()

    # Configurazione del logger
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Percorso alla directory del file corrente
    file_dir = dirname(__file__)
    stele_path = join(file_dir, arguments.data_path)

    # Lista di tutte le steli (cartelle) nella directory specificata
    steles = [join(stele_path, f) for f in listdir(stele_path) if isdir(join(stele_path, f))]

    res_image_paths, labels = [], []

    # Itera su ciascuna stele per ottenere i percorsi delle immagini e le etichette
    for stele in steles:
        image_paths = [join(stele, f) for f in listdir(stele) if isfile(join(stele, f))]

        for path in image_paths:
            res_image_paths.append(path)
            labels.append(path[(path.rfind("_") + 1): path.rfind(".")])

    list_of_paths = np.asarray(res_image_paths)
    labels = np.array(labels)

    logging.debug(f"Numero totale di etichette: {len(set(labels))}")

    # Identifica le etichette che appaiono solo una volta nel dataset
    labels_just_once = np.array([l for (l, c) in Counter(labels).items() if c <= 1])
    logging.debug(f"Etichette che appaiono solo una volta: {len(labels_just_once)}")

    # Le etichette che appaiono una sola volta saranno aggiunte solo al set di addestramento
    to_be_added_to_train_only = np.nonzero(np.isin(labels, labels_just_once))[0]

    # Le etichette sconosciute devono essere rimosse dal dataset
    to_be_deleted = np.nonzero(labels == UNKNOWN_LABEL)[0]

    # Rimuove gli elementi che devono essere aggiunti solo al set di addestramento o che sono sconosciuti
    to_be_deleted = np.concatenate([to_be_deleted, to_be_added_to_train_only])
    filtered_list_of_paths = np.delete(list_of_paths, to_be_deleted, 0)
    filtered_labels = np.delete(labels, to_be_deleted, 0)

    # Suddivisione del dataset in set di addestramento e test
    train_paths, test_paths, y_train, y_test = train_test_split(filtered_list_of_paths,
                                                                filtered_labels,
                                                                stratify=filtered_labels,
                                                                test_size=arguments.test_fraction,
                                                                random_state=arguments.seed)

    # Aggiunge al set di addestramento le etichette che appaiono solo una volta
    train_paths = np.concatenate([train_paths, list_of_paths[to_be_added_to_train_only]])
    y_train = np.concatenate([y_train, labels[to_be_added_to_train_only]])

    # Creazione delle directory per i dati preparati
    makedirs(arguments.prepared_data_path, exist_ok=True)
    [makedirs(join(arguments.prepared_data_path, "train", l), exist_ok=True) for l in set(y_train)]
    [makedirs(join(arguments.prepared_data_path, "test", l), exist_ok=True) for l in set(y_test)]

    # Copia le immagini nei rispettivi set di addestramento e test
    for fp, label in zip(train_paths, y_train):
        fn = join(arguments.prepared_data_path, "train", label, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)

    for fp, label in zip(test_paths, y_test):
        fn = join(arguments.prepared_data_path, "test", label, hashlib.md5(fp.encode('utf-8')).hexdigest() + ".png")
        shutil.copyfile(fp, fn)
