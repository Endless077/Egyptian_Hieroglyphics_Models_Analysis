import subprocess
from codecarbon import EmissionsTracker


def train_yolov5(data_yaml, epochs=50, batch_size=16, img_size=640):
    """
    Funzione per addestrare un modello YOLOv5 utilizzando uno script di addestramento.

    Args:
        data_yaml (str): Il percorso al file YAML contenente la configurazione dei dati per l'addestramento.
        epochs (int): Il numero di epoche per cui addestrare il modello (default: 50).
        batch_size (int): La dimensione del batch utilizzata durante l'addestramento (default: 16).
        img_size (int): La dimensione delle immagini di input (default: 640).
    """
    # Costruisce il comando per l'addestramento di YOLOv5
    command = f"python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights yolov5s.pt --cache"

    # Esegue il comando tramite il terminale
    subprocess.run(command, shell=True)


if __name__ == "__main__":
    # Avvia il monitoraggio delle emissioni di CO2 con CodeCarbon
    tracker = EmissionsTracker()  # È possibile specificare una directory di output se necessario
    tracker.start()

    # Addestra il modello YOLOv5
    train_yolov5(data_yaml='./configs/data.yaml', epochs=100, batch_size=4, img_size=320)

    # Ferma il monitoraggio delle emissioni di CO2
    tracker.stop()
