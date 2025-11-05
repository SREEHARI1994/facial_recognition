from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
from simpleface.recognizer import SimpleFaceRecognizer

# Load dataset
lfw = fetch_lfw_people(min_faces_per_person=10, resize=0.5, color=True)
X_train, X_test, y_train, y_test = train_test_split(
    lfw.images, lfw.target, test_size=0.3, stratify=lfw.target, random_state=42
)

recognizer = SimpleFaceRecognizer()

# --- Training phase ---
print("[INFO] Adding training faces...")
for i in range(len(X_train)):
    name = lfw.target_names[y_train[i]]
    img = (X_train[i] * 255).astype("uint8")
    recognizer.add_person(name, img)


# --- Testing phase ---
print("[INFO] Testing on unseen faces...")
correct = 0
total = 0

for i, img in enumerate(X_test):
    true_name = lfw.target_names[y_test[i]]
    img = (img * 255).astype("uint8")

    # Use your recognizer directly
    results = recognizer.recognize_image(img)

    predicted_name = results[0]["name"] if results else None
    if predicted_name == true_name:
        correct += 1
    total += 1

accuracy = correct / total
print(f"\n✅ Recognition Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
