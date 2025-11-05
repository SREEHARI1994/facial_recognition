from .models_utils import ensure_model_exists

def main():
    print("[INFO] Setting up facial_recognition model...")
    ensure_model_exists()
    print("[INFO] Setup complete. You’re ready to use facial_recognition!")
