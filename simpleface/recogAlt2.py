import cv2
import numpy as np
import onnxruntime as ort
import json
import os
from datetime import datetime


class SimpleFaceRecognizer:
    def __init__(self, model_path=None, db_path=None):
        # --- Embedding Model (Glint360K / InsightFace r100) ---
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "glintr100.onnx")
        self.model_path = model_path

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.sess = ort.InferenceSession(self.model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name

        # --- RetinaFace Detector ---
        det_model_path = os.path.join(os.path.dirname(__file__), "retinaface.onnx")
        if not os.path.exists(det_model_path):
            raise FileNotFoundError(f"RetinaFace model not found: {det_model_path}")

        self.det_sess = ort.InferenceSession(det_model_path, providers=["CPUExecutionProvider"])
        self.det_input = self.det_sess.get_inputs()[0].name

        # --- Face Database Path ---
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "face_db.json")
        self.db_path = db_path

        self.known_faces = {}
        self.load_database()

    # ---------- Internal helpers ----------

    def _preprocess(self, img):
        """Preprocess face crop for Glint360K r100 model (NCHW)."""
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)
        return img

    def _cosine_similarity(self, emb1, emb2):
        """Cosine similarity (dot product for normalized vectors)."""
        return np.dot(emb1, emb2)

    # ---------- RetinaFace Detection ----------

    def detect_faces(self, image):

        # Resize to (640, 608) because model expects that
        img = cv2.resize(image, (640, 608))  # (width, height)

        # Convert to float32 and normalize
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0

        # Expand dimensions to (1, 608, 640, 3) for NHWC
        blob = np.expand_dims(img, axis=0)

        # Run model
        outputs = self.det_sess.run(None, {self.det_input: blob})

        # Unpack outputs
        bboxes, confs = outputs[0], outputs[1]
        print(f"bboxes: {bboxes.shape} confs: {confs.shape}")

        # Remove batch dimension
        bboxes = np.squeeze(bboxes, axis=0)
        confs = np.squeeze(confs, axis=0)

        faces = []
        for i in range(len(bboxes)):
            conf = float(confs[i][1]) if len(confs[i]) > 1 else float(confs[i][0])
            if conf < 0.8:
                continue
            x1, y1, x2, y2 = bboxes[i]
            faces.append((x1, y1, x2, y2, conf))

        print(f"Detected {len(faces)} faces with confidence >= 0.8")
        return faces



    # ---------- Embedding + Matching ----------

    def embed(self, face_crop):
        x = self._preprocess(face_crop)
        emb = self.sess.run(None, {self.input_name: x})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb

    def _match_embedding(self, emb, threshold):
        best_name, best_score = "Unknown", -1.0

        if not self.known_faces:
            return best_name, 0.0

        for name, known_emb in self.known_faces.items():
            score = self._cosine_similarity(emb, known_emb)
            if score > best_score:
                best_score, best_name = score, name
            print(f"[DEBUG] Comparing with {name}: {score:.4f}")

        if best_score < threshold:
            best_name = "Unknown"

        return best_name, best_score

    # ---------- Database Handling ----------

    def load_database(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.known_faces = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
                print(f"[INFO] Loaded {len(self.known_faces)} known faces from {self.db_path}")
            except json.JSONDecodeError:
                print(f"[WARN] Could not decode JSON from {self.db_path}. Starting empty DB.")
                self.known_faces = {}
        else:
            self.known_faces = {}

    def save_database(self):
        with open(self.db_path, "w") as f:
            json.dump({k: v.tolist() for k, v in self.known_faces.items()}, f)
        print(f"[INFO] Saved {len(self.known_faces)} faces to {self.db_path}")

    # ---------- Face Registration ----------

    def add_person(self, name, image_path):
        detections = self.recognize_file(image_path, return_embeddings=True, save_output=False)
        if not detections:
            print(f"[WARN] No face found in {image_path}")
            return

        emb = detections[0]["embedding"]
        self.known_faces[name] = emb
        self.save_database()
        print(f"[INFO] Added {name} to database")

    # ---------- Recognition ----------

    def recognize_file(self, image_path, threshold=0.8, save_output=True, return_embeddings=False):
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        faces = self.detect_faces(image)
        results = []
        h, w = image.shape[:2]

        for f in faces:
            x, y, bw, bh = f["box"]
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            crop = image[max(0, y1):min(y2, h), max(0, x1):min(x2, w)]

            if crop.size == 0:
                continue

            emb = self.embed(crop)
            recognized_name, best_score = self._match_embedding(emb, threshold)

            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)
            label = f"{recognized_name} ({best_score:.2f})"
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            result = {
                "name": recognized_name,
                "score": best_score,
                "confidence": f["confidence"],
                "box": f["box"]
            }
            if return_embeddings:
                result["embedding"] = emb
            results.append(result)

        if save_output:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("output", f"recognized_{ts}.jpg")
            cv2.imwrite(out_path, image)
            print(f"[INFO] Saved annotated image to {out_path}")

        return results

    # ---------- Visualization Helper ----------

    def visualize_detections(self, image_path, detections, save_to="debug_faces.jpg"):
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w = img.shape[:2]
        for det in detections:
            x, y, bw, bh = det["box"]
            conf = det.get("confidence", 0)
            x1, y1 = int(x * w), int(y * h)
            x2, y2 = int((x + bw) * w), int((y + bh) * h)
            color = (0, 255, 0) if conf > 0.7 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{conf:.2f}", (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imwrite(save_to, img)
        print(f"Saved visualization to {save_to}")
