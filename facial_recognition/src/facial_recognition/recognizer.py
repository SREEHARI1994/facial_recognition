import cv2
import numpy as np
import onnxruntime as ort
import mediapipe as mp
import json
import os
from datetime import datetime
from .models_utils import ensure_model_exists

# --- Configuration ---
# Optimal threshold for Cosine Similarity with ArcFace.
# HIGHER value means closer match (0.5 to 0.6 is common).
RECOGNITION_THRESHOLD = 0.3



class SimpleFaceRecognizer:
    def __init__(self, model_path=None, db_path=None):
        
        if model_path is None:
            model_path = ensure_model_exists()
        self.model_path = model_path  # ✅ assign before using it
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # ONNX Runtime setup
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name

        # Initialize Mediapipe face detector
        #self.detector = mp.solutions.face_detection.FaceDetection(
            #odel_selection=1, min_detection_confidence=0.5
        #)
        #self.detector = mp.solutions.face_detection.FaceDetection(
            #model_selection=0,  # short-range, works better for small faces
            #min_detection_confidence=0.3
        #)
        # --- Face database path ---
        if db_path is None:
            db_path = os.path.join(os.path.dirname(__file__), "face_db.json")
        self.db_path = db_path

        self.known_faces = {}
        self.load_database()

    # ---------- Internal helpers ----------

    def _preprocess(self, img):
        """Preprocess face crop for ArcFace model (TensorFlow-style NHWC)."""
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0  # normalize
        img = np.expand_dims(img, axis=0)  # (1, 112, 112, 3)
        return img

    def _cosine_similarity(self, emb1, emb2):
        """Calculates Cosine Similarity (Dot Product for normalized vectors)."""
        return np.dot(emb1, emb2)

    # ---------- Core methods ----------

    def detect_faces(self, image):
        """Detect faces using Mediapipe with automatic model selection (short vs full range)."""

        # --- 1. Choose the right Mediapipe model ---
        h, w = image.shape[:2]
        if max(h, w) < 150:
            model_selection = 0  # short-range model for small/cropped faces
            min_conf = 0.3
        else:
            model_selection = 1  # full-range model for normal camera photos
            min_conf = 0.5
        print(f"[DEBUG] Using Mediapipe model={model_selection} (h={h}, w={w})")

        # --- 2. Initialize detectors lazily (and cache them) ---
        if not hasattr(self, "_detectors"):
            self._detectors = {}

        if model_selection not in self._detectors:
            self._detectors[model_selection] = mp.solutions.face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_conf
            )

        detector = self._detectors[model_selection]

        # --- 3. Run detection ---
        results = detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        detections = []
        if not results.detections:
            return detections

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            conf = det.score[0] if det.score else 0
            detections.append({
                "box": (bbox.xmin, bbox.ymin, bbox.width, bbox.height),
                "confidence": conf
            })

        # --- 4. Filter weak detections ---
        detections = [d for d in detections if d["confidence"] > min_conf]
        return detections


    def embed(self, face_crop):
        """Generate normalized embedding for cropped face."""
        x = self._preprocess(face_crop)
        emb = self.sess.run(None, {self.input_name: x})[0][0]
        emb = emb / np.linalg.norm(emb)
        return emb

    def _match_embedding(self, emb, threshold=RECOGNITION_THRESHOLD):
        """Find best match among known faces using cosine similarity."""
        best_name = "Unknown"
        best_score = -1.0

        if not self.known_faces:
            return "Unknown", 0.0

        for name, known_emb in self.known_faces.items():
            score = self._cosine_similarity(emb, known_emb)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score < threshold:
            best_name = "Unknown"

        return best_name, best_score

    # ---------- Database Handling ----------

    def load_database(self):
        """Load known face embeddings from disk."""
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    data = json.load(f)
                self.known_faces = {
                    k: np.array(v, dtype=np.float32) for k, v in data.items()
                }
                print(f"[INFO] Loaded {len(self.known_faces)} known faces from {self.db_path}")
            except json.JSONDecodeError:
                print(f"[WARN] Could not decode JSON from {self.db_path}. Starting with empty database.")
                self.known_faces = {}
        else:
            self.known_faces = {}

    def save_database(self):
        """Save known face embeddings to disk."""
        with open(self.db_path, "w") as f:
            json.dump({k: v.tolist() for k, v in self.known_faces.items()}, f)
        print(f"[INFO] Saved {len(self.known_faces)} faces to {self.db_path}")

    # ---------- Face Registration ----------

    def add_person(self, name, image):
        """Add a new person to the known faces database."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                print(f"[WARN] Could not read {image}")
                return
        elif isinstance(image, np.ndarray):
            img = image
        else:
            raise ValueError("`image` must be a file path (str) or a NumPy array.")

        # Convert grayscale → BGR if needed
        #if len(img.shape) == 2:
            #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        detections = self.recognize_image(img, return_embeddings=True, save_output=False)
        if not detections:
            print(f"[WARN] No face found (image shape={img.shape})")


            return

        emb = detections[0]["embedding"]
        self.known_faces[name] = emb
        self.save_database()
        print(f"[INFO] Added {name} to database")

    
    
    def recognize_image(self, input_image, threshold=RECOGNITION_THRESHOLD, save_output=True, return_embeddings=False):
        """Detect faces, recognize them, draw boxes, and optionally save annotated image."""
        # Determine if `image` is a file path or an image array
        if isinstance(input_image, str):
            if not os.path.exists(input_image):
                raise FileNotFoundError(f"Image file not found: {input_image}")
            image = cv2.imread(input_image)
            if image is None:
                raise ValueError(f"Unable to read image file: {input_image}")
            
        elif isinstance(input_image, np.ndarray):
            image = input_image
        else:
            raise TypeError("Input must be either a NumPy array or a valid image file path.")

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
            recognized_name, best_score = self._match_embedding(emb, threshold=threshold)

            color = (0, 255, 0) if recognized_name != "Unknown" else (0, 0, 255)

            result = {
                "name": recognized_name,
                "score": best_score,
                "confidence": f["confidence"],
                "box": f["box"]
            }
            if return_embeddings:
                result["embedding"] = emb
            results.append(result)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{recognized_name} ({best_score:.2f})"
            cv2.putText(image, label, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        if save_output:
            os.makedirs("output", exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("output", f"recognized_{ts}.jpg")
            cv2.imwrite(out_path, image)
            print(f"[INFO] Saved annotated image to {out_path}")
            
        return results
    
    
    
    '''
    def visualize_detections(self, image_path, detections, save_to="debug_faces.jpg"):
        """Draw bounding boxes for detected faces."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h, w = img.shape[:2]
        for i, det in enumerate(detections):
            x, y, bw, bh = det["box"]
            conf = det.get("confidence", 0)
            x1 = int(x * w)
            y1 = int(y * h)
            x2 = int((x + bw) * w)
            y2 = int((x + bh) * h)
            color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                f"{conf:.2f}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        cv2.imwrite(save_to, img)
        print(f"Saved visualization to {save_to}")

    '''