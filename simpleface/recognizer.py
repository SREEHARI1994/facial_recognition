import os
import cv2
import numpy as np
import onnxruntime as ort
from numpy.linalg import norm
import datetime


class SimpleFaceRecognizer:
    def __init__(self, det_model=None, emb_model=None):
        # Load face detection model
        print("[INFO] Loading RetinaFace ONNX model...")
        if det_model is None:
            det_model = os.path.join(os.path.dirname(__file__), "retinaface.onnx")
        self.det_model = det_model

        if not os.path.exists(self.det_model):
            raise FileNotFoundError(f"Detection Model file not found: {self.det_model}")

        if emb_model is None:
            emb_model = os.path.join(os.path.dirname(__file__), "glintr100.onnx")
        self.emb_model = emb_model

        if not os.path.exists(self.emb_model):
            raise FileNotFoundError(f"Embedding Model file not found: {self.emb_model}")

        self.det_sess = ort.InferenceSession(det_model, providers=["CPUExecutionProvider"])
        self.det_input_name = self.det_sess.get_inputs()[0].name

        # Load face embedding model
        print("[INFO] Loading Glint100 embedding model...")
        self.emb_sess = ort.InferenceSession(emb_model, providers=["CPUExecutionProvider"])
        self.emb_input_name = self.emb_sess.get_inputs()[0].name

        # Load database
        self.db_path = "simpleface/face_db.npz"
        self.database = {}
        if os.path.exists(self.db_path):
            data = np.load(self.db_path, allow_pickle=True)
            self.database = data["faces"].item()
            print(f"[INFO] Loaded {len(self.database)} people from database.")

    # ------------------------------------------------------------
    def preprocess_retina(self, img):
        """Resize and prepare image for RetinaFace."""
        img_resized = cv2.resize(img, (640, 608))
        blob = img_resized.astype(np.float32)
        blob = np.expand_dims(blob, axis=0)  # NHWC format
        return blob, img_resized.shape[1], img_resized.shape[0]

    # ------------------------------------------------------------
    def detect_faces(self, img, conf_threshold=0.6):
        """Run RetinaFace ONNX detector manually."""
        blob, w, h = self.preprocess_retina(img)
        outputs = self.det_sess.run(None, {self.det_input_name: blob})

        # Expected: bboxes and confs from the model
        bboxes, confs = outputs[0], outputs[1]
        print(f"bboxes: {bboxes.shape} confs: {confs.shape}")

        bboxes = bboxes[0]
        confs = confs[0]

        faces = []
        img_h, img_w = img.shape[:2]

        for i in range(len(bboxes)):
            conf = confs[i][-1] if confs[i].ndim > 0 else confs[i]
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = bboxes[i]

            # Convert normalized coords to pixels if needed
            if x2 <= 1.0 and y2 <= 1.0:
                x1 *= img_w
                y1 *= img_h
                x2 *= img_w
                y2 *= img_h

            # Clip safely
            x1 = int(max(0, min(x1, img_w - 1)))
            y1 = int(max(0, min(y1, img_h - 1)))
            x2 = int(max(0, min(x2, img_w - 1)))
            y2 = int(max(0, min(y2, img_h - 1)))

            if x2 <= x1 or y2 <= y1:
                continue

            faces.append((x1, y1, x2, y2, conf))

        print(f"Detected {len(faces)} faces with confidence >= {conf_threshold}")
        return faces

    # ------------------------------------------------------------
    def get_embedding(self, face_img):
        """Compute 512D embedding for a cropped face."""
        if face_img is None or face_img.size == 0:
            return None
        face = cv2.resize(face_img, (112, 112))
        face = face.astype(np.float32) / 127.5 - 1.0
        face = np.expand_dims(np.transpose(face, (2, 0, 1)), axis=0)
        emb = self.emb_sess.run(None, {self.emb_input_name: face})[0]
        emb = emb / norm(emb)
        return emb.flatten()

    # ------------------------------------------------------------
    def cosine_similarity(self, a, b):
        """Compute cosine similarity safely between two vectors."""
        if a is None or b is None:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

    # ------------------------------------------------------------
    def recognize_file(self, image_path, threshold=0.75, return_embeddings=False, save_output=False):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Could not read image: {image_path}")
            return []

        faces = self.detect_faces(img)
        results = []

        for (x1, y1, x2, y2, conf) in faces:
            face_crop = img[y1:y2, x1:x2]
            emb = self.get_embedding(face_crop)
            if emb is None:
                continue

            best_name, best_score = "Unknown", 0.0
            for name, db_emb in self.database.items():
                score = self.cosine_similarity(emb, db_emb)
                if score > best_score:
                    best_name, best_score = name, score

            match = best_name if best_score >= threshold else "Unknown"

            results.append({
                "name": match,
                "confidence": float(conf),
                "score": float(best_score),
                "bbox": (x1, y1, x2, y2),
            })

        # Save visualization if requested
        if save_output:
            detections_for_draw = []
            for r in results:
                detections_for_draw.append((
                    r["bbox"][0], r["bbox"][1], r["bbox"][2], r["bbox"][3],
                    r["confidence"], r["name"], r["score"]
                ))

            # Even if no faces are found, still save original image
            visualize_detections(img, detections_for_draw, save_path="output")

        return results

    # ------------------------------------------------------------
    def add_person(self, name, image_path, conf_threshold=0.6):
        img = cv2.imread(image_path)
        if img is None:
            print(f"[ERROR] Cannot read {image_path}")
            return

        faces = self.detect_faces(img, conf_threshold=conf_threshold)
        if not faces:
            print(f"[WARN] No face found in {image_path}")
            return

        # Take the face with the highest confidence
        faces.sort(key=lambda f: f[4], reverse=True)
        (x1, y1, x2, y2, conf) = faces[0]
        face_crop = img[y1:y2, x1:x2]
        emb = self.get_embedding(face_crop)

        if emb is None:
            print(f"[ERROR] Could not extract embedding for {name}.")
            return

        self.database[name] = emb
        np.savez(self.db_path, faces=self.database)
        print(f"[INFO] Added '{name}' to database (conf={conf:.2f}).")


# ----------------------------------------------------------------
def visualize_detections(image, detections, save_path=None):
    """Draw boxes and names on an image."""
    for (x1, y1, x2, y2, conf, name, score) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{name} ({score:.2f})"
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(save_path, f"recognized_{ts}.jpg")
        cv2.imwrite(out_path, image)
        print(f"[INFO] Saved annotated image to {out_path}")
