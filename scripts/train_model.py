import os
import cv2
import pickle
import numpy as np

from backend.detector import HandDetector
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

detector = HandDetector()

data = []
labels = []

dataset_path = "dataset"

for label in os.listdir(dataset_path):

    folder_path = os.path.join(dataset_path, label)

    if not os.path.isdir(folder_path):
        continue

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        img = cv2.imread(image_path)

        if img is None:
            continue

        img, landmarks = detector.detect_hands(img)

        # Keep only valid hand detections
        if landmarks and len(landmarks) == 21:

            flat = np.array(landmarks).flatten()

            # 21 landmarks × 2 values (x,y) = 42
            if len(flat) == 42:

                data.append(flat)
                labels.append(label)

print("Dataset Loaded")
print("Samples:", len(data))

X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Accuracy:", accuracy)

with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully")