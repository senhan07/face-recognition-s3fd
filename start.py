import os
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from tqdm import tqdm

def detect_faces(image):
    mtcnn = MTCNN()
    boxes, _ = mtcnn.detect(image)
    return boxes if boxes is not None else []

def filter_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filenames = sorted(os.listdir(input_folder))
    num_images = len(filenames)
    progress_bar = tqdm(total=num_images, desc='Filtering Images')

    face_dict = {}
    person_index = 0

    for filename in filenames:
        progress_bar.set_postfix_str(filename)
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        boxes = detect_faces(image)
        num_faces = len(boxes)

        labels = []

        for face in boxes:
            face = tuple(face.tolist())  # Convert to tuple

            if face not in face_dict:
                face_dict[face] = person_index
                person_index += 1

            labels.append(str(face_dict[face]))

        if num_faces > 0:
            label = labels[0]
        else:
            label = "unknown"

        output_filename = f'{os.path.splitext(filename)[0]}_{label}.jpg'
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, image)

        progress_bar.update(1)

    progress_bar.close()
    print('Filtering complete!')

# Usage example
input_folder = 'input'
output_folder = 'output'
filter_images(input_folder, output_folder)
