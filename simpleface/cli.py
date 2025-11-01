import argparse
from .recognizer import SimpleFaceRecognizer

def main():
    parser = argparse.ArgumentParser(description="SimpleFace CLI Tool")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cosine similarity threshold for recognition (default: 0.5)",
    )
    args = parser.parse_args()

    # Initialize recognizer (loads model + database)
    recognizer = SimpleFaceRecognizer()

    print("[INFO] Running face recognition...")
    results = recognizer.recognize_file(args.image, threshold=args.threshold)

    if not results:
        print("[WARN] No faces detected.")
        return

    # Print results to console
    print("\n=== Recognition Results ===")
    for i, r in enumerate(results):
        print(
            f"Face {i+1}: {r['name']} (confidence={r['confidence']:.2f}, match_score={r['score']:.3f})"
        )

if __name__ == "__main__":
    main()
