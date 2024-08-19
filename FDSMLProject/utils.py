import os
import cv2

# Definisci class_to_idx e idx_to_class
class_to_idx = {
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


# Funzione per convertire il dataset YOLOv5 in un dataset di classificazione
def convert_yolo_to_classification(dataset_dir='dataset', output_dir='classification_dataset'):
    # Funzione per convertire coordinate normalizzate in coordinate di pixel
    def normalize_to_pixel_coords(coords, img_width, img_height):
        pixel_coords = []
        for i in range(0, len(coords), 2):
            x = int(coords[i] * img_width)
            y = int(coords[i + 1] * img_height)
            pixel_coords.append((x, y))
        return pixel_coords

    # Funzione per ottenere la bounding box minima da una serie di punti
    def get_bounding_box_from_points(points):
        x_min = min(point[0] for point in points)
        y_min = min(point[1] for point in points)
        x_max = max(point[0] for point in points)
        y_max = max(point[1] for point in points)
        return x_min, y_min, x_max, y_max

    # Processare ogni parte del dataset (train, val, test)
    for split in ['train', 'valid', 'test']:
        split_images_dir = os.path.join(dataset_dir, split, 'images')
        split_labels_dir = os.path.join(dataset_dir, split, 'labels')
        split_output_dir = os.path.join(output_dir, split)

        # Creare la struttura di output
        if not os.path.exists(split_output_dir):
            os.makedirs(split_output_dir)

        # Processare ogni immagine e file di annotazione
        for label_file in os.listdir(split_labels_dir):
            with open(os.path.join(split_labels_dir, label_file), 'r') as f:
                annotations = f.readlines()

            image_path = os.path.join(split_images_dir, label_file.replace('.txt', '.jpg'))
            image = cv2.imread(image_path)
            img_height, img_width = image.shape[:2]

            for annotation in annotations:
                values = list(map(float, annotation.split()))

                class_id = int(values[0])
                coords = values[1:]  # Tutte le coppie di coordinate

                if len(coords) % 2 != 0:
                    print(f"Errore: numero dispari di valori nelle coordinate: {coords}")
                    continue

                # Converti le coordinate normalizzate in coordinate di pixel
                points = normalize_to_pixel_coords(coords, img_width, img_height)

                # Ottieni la bounding box che contiene tutti i punti
                x_min, y_min, x_max, y_max = get_bounding_box_from_points(points)

                # Ritagliare l'immagine
                cropped_img = image[y_min:y_max, x_min:x_max]

                # Salva l'immagine ritagliata nella cartella della classe corrispondente
                class_folder = os.path.join(split_output_dir, str(class_id))
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                img_name = os.path.basename(image_path).replace('.jpg', f'_{x_min}_{y_min}.jpg')
                cv2.imwrite(os.path.join(class_folder, img_name), cropped_img)

    print("Conversione completata!")
