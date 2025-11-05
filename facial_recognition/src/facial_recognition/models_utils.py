import os
import urllib.request

MODEL_URL = "https://huggingface.co/yourusername/simpleface-models/resolve/main/arcface.onnx"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "arcface.onnx")

def ensure_model_exists():
    """Download the ArcFace model if it doesn't exist locally."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH

    print("[INFO] ArcFace model not found. Downloading...")
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model downloaded successfully to {MODEL_PATH}")
    except Exception as e:
        raise RuntimeError(f"Failed to download model: {e}")

    return MODEL_PATH
