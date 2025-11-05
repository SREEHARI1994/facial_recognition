#r = SimpleFaceRecognizer()
#r.add_person("Sruthi", "simpleface/faces/sruthi.jpg")
#r.add_person("Sreehari", "simpleface/faces/sreehari.jpg")
import os
import cv2
from .recognizer import SimpleFaceRecognizer

def add_faces_from_folder(folder_path):
    recognizer = SimpleFaceRecognizer()

    print(f"[INFO] Scanning folder: {folder_path}")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("[WARN] No image files found in this folder.")
        return

    for filename in image_files:
        name = os.path.splitext(filename)[0]
        file_path = os.path.join(folder_path, filename)

        print(f"[ADD] Adding {name} from {file_path}")
        try:
            recognizer.add_person(name, file_path)
        except Exception as e:
            print(f"[ERROR] Failed to add {name}: {e}")

    print("[INFO] Face database updated successfully.")
