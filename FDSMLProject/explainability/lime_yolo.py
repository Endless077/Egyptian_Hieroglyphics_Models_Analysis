import warnings

# Disabilita i warning per mantenere l'output pulito
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_cam_image
from PIL import Image, UnidentifiedImageError

# Colori casuali per le classi (utilizzati per disegnare le bounding boxes)
COLORS = np.random.uniform(0, 255, size=(80, 3))


def parse_detections(results):
    """
    Estrae le bounding boxes, i colori e i nomi delle classi dai risultati del modello YOLOv5.

    Args:
        results (object): Risultati dell'inferenza del modello YOLOv5.

    Returns:
        Tuple[List[Tuple[int, int, int, int]], List[np.ndarray], List[str]]:
        - boxes: Coordinate delle bounding boxes.
        - colors: Colori associati a ciascuna classe.
        - names: Nomi delle classi rilevate.
    """
    # Estrae le informazioni dalle predizioni del modello YOLOv5
    detections = results.pandas().xyxy[0]
    detections = detections.to_dict()
    boxes, colors, names = [], [], []

    for i in range(len(detections["xmin"])):
        confidence = detections["confidence"][i]
        if confidence < 0.2:
            continue  # Ignora le predizioni con confidenza bassa
        xmin = int(detections["xmin"][i])
        ymin = int(detections["ymin"][i])
        xmax = int(detections["xmax"][i])
        ymax = int(detections["ymax"][i])
        name = detections["name"][i]
        category = int(detections["class"][i])
        color = COLORS[category]

        # Salva le informazioni della bounding box, il colore e il nome della classe
        boxes.append((xmin, ymin, xmax, ymax))
        colors.append(color)
        names.append(name)
    return boxes, colors, names


def draw_detections(boxes, colors, names, img):
    """
    Disegna le bounding boxes e i nomi delle classi sull'immagine.

    Args:
        boxes (List[Tuple[int, int, int, int]]): Coordinate delle bounding boxes.
        colors (List[np.ndarray]): Colori associati a ciascuna classe.
        names (List[str]): Nomi delle classi rilevate.
        img (np.ndarray): Immagine su cui disegnare le bounding boxes.

    Returns:
        np.ndarray: Immagine con le bounding boxes e i nomi delle classi disegnati.
    """
    for box, color, name in zip(boxes, colors, names):
        xmin, ymin, xmax, ymax = box
        # Disegna il rettangolo della bounding box
        cv2.rectangle(
            img,
            (xmin, ymin),
            (xmax, ymax),
            color,
            2)

        # Aggiungi il nome della classe sopra la bounding box
        cv2.putText(img, name, (xmin, ymin - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                    lineType=cv2.LINE_AA)
    return img


def renormalize_cam_in_bounding_boxes(boxes, colors, names, image_float_np, grayscale_cam):
    """
    Rinormalizza la CAM all'interno delle bounding boxes e disegna le bounding boxes sull'immagine.

    Args:
        boxes (List[Tuple[int, int, int, int]]): Coordinate delle bounding boxes.
        colors (List[np.ndarray]): Colori associati a ciascuna classe.
        names (List[str]): Nomi delle classi rilevate.
        image_float_np (np.ndarray): Immagine normalizzata in float.
        grayscale_cam (np.ndarray): CAM in scala di grigi.

    Returns:
        np.ndarray: Immagine con la CAM rinormalizzata e le bounding boxes disegnate.
    """
    # Crea una CAM rinormalizzata all'interno delle bounding boxes
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.float32)
    for x1, y1, x2, y2 in boxes:
        # Rinormalizza la CAM all'interno della bounding box
        renormalized_cam[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    # Disegna le bounding boxes sull'immagine con la CAM rinormalizzata
    image_with_bounding_boxes = draw_detections(boxes, colors, names, eigencam_image_renormalized)
    return image_with_bounding_boxes


# Specifica il percorso dell'immagine personalizzata
image_path = "../datasets/dataset_yolo/train/images/Screen-Shot-2020-07-06-at-4-08-53-PM_0_png.rf.93d6504cc1c64701273f399044eecf4d.jpg"

# Carica l'immagine e gestisci eventuali errori
try:
    img = np.array(Image.open(image_path))
except UnidentifiedImageError as e:
    print(f"Failed to load image from path. Error: {e}")
    exit(1)

# Ridimensiona l'immagine a 640x640 pixel
img = cv2.resize(img, (640, 640))
rgb_img = img.copy()  # Crea una copia dell'immagine in RGB
img = np.float32(img) / 255  # Normalizza l'immagine
transform = transforms.ToTensor()
tensor = transform(img).unsqueeze(0)  # Converte l'immagine in un tensore e aggiunge una dimensione per il batch

# Carica il modello YOLOv5 addestrato
model_path = "../results/results_yolov5/best_weights/best.pt"
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
model.eval()  # Imposta il modello in modalità di valutazione
model.cpu()  # Sposta il modello sulla CPU
target_layers = [model.model.model.model[-2]]  # Specifica i livelli target per la CAM

# Esegui l'inferenza sull'immagine personalizzata
results = model([rgb_img])
boxes, colors, names = parse_detections(results)
detections = draw_detections(boxes, colors, names, rgb_img.copy())

# Genera la CAM (Class Activation Map) utilizzando EigenCAM
cam = EigenCAM(model, target_layers)
grayscale_cam = cam(tensor)[0, :, :]  # Ottieni la CAM come immagine in scala di grigi
cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)

# Mostra l'immagine della CAM
cam_image_pil = Image.fromarray(cam_image)
cam_image_pil.show()  # Questo visualizzerà l'immagine della CAM

# Salva l'immagine della CAM se necessario
cam_image_pil.save("cam_image_custom.png")

# Rinormalizza la CAM all'interno delle bounding boxes e visualizzala
renormalized_cam_image = renormalize_cam_in_bounding_boxes(boxes, colors, names, img, grayscale_cam)

# Mostra l'immagine della CAM rinormalizzata
renormalized_cam_image_pil = Image.fromarray(renormalized_cam_image)
renormalized_cam_image_pil.show()  # Questo visualizzerà l'immagine della CAM rinormalizzata

# Salva l'immagine della CAM rinormalizzata se necessario
renormalized_cam_image_pil.save("renormalized_cam_image_custom.png")

# Combina e mostra tutte le immagini affiancate
combined_image = np.hstack((rgb_img, cam_image, renormalized_cam_image))
combined_image_pil = Image.fromarray(combined_image)
combined_image_pil.show()  # Questo visualizzerà l'immagine combinata

# Salva l'immagine combinata se necessario
combined_image_pil.save("combined_image_custom.png")
