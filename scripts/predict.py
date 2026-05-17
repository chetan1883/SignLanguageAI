import cv2
import pickle
import numpy as np
import pyttsx3

from collections import Counter
from backend.detector import HandDetector

# Load trained model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Voice engine
engine = pyttsx3.init()

cap = cv2.VideoCapture(0)

detector = HandDetector()

# Variables
word = ""
predictions = []

stable_prediction = ""

while True:

    success, img = cap.read()

    if not success:
        break

    img, landmarks = detector.detect_hands(img)

    if landmarks and len(landmarks) == 21:

        flat = np.array(landmarks).flatten()

        if len(flat) == 42:

            prediction = model.predict([flat])[0]

            # Store predictions
            predictions.append(prediction)

            # Keep only last 15 predictions
            if len(predictions) > 15:
                predictions.pop(0)

            # Most common prediction
            stable_prediction = Counter(predictions).most_common(1)[0][0]

            # Display prediction
            cv2.putText(
                img,
                f"Prediction: {stable_prediction}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # Display formed word
    cv2.putText(
        img,
        f"Word: {word}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
        2
    )

    cv2.imshow("Sign Language AI", img)

    key = cv2.waitKey(1)

    # SPACE → add stable letter
    if key == 32:

        if stable_prediction != "":
            word += stable_prediction

    # BACKSPACE → remove last letter
    elif key == 8:
        word = word[:-1]

    # C → clear word
    elif key == ord("c"):
        word = ""

    # V → speak word
    elif key == ord("v"):

        if word != "":
            engine.say(word)
            engine.runAndWait()

    # ESC → exit
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()