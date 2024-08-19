import subprocess

from codecarbon import EmissionsTracker


# Funzione per addestrare YOLOv5
def train_yolov5(data_yaml, epochs=50, batch_size=16, img_size=640):
    # Comando per l'addestramento
    command = f"python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights yolov5s.pt --cache"

    # Esegui il comando
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Avvio del tracker di CodeCarbon
    tracker = EmissionsTracker()  # Puoi specificare una directory di output se necessario
    tracker.start()

    # Addestramento del modello YOLOv5
    train_yolov5(data_yaml='./configs/data.yaml', epochs=80, batch_size=4, img_size=320)

    tracker.stop()

    # Conversione del dataset YOLOv5 in dataset di classificazione
    # convert_yolo_to_classification(dataset_dir='dataset', output_dir='classification_dataset')
