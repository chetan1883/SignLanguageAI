import cv2
import os
from backend.detector import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector()

label = "Z"
save_path = f"dataset/{label}"

os.makedirs(save_path, exist_ok=True)

count = len(os.listdir(save_path))

while True:
    success, img = cap.read()
    img, landmarks = detector.detect_hands(img)

    cv2.putText(img, f"Label: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.putText(img, f"Saved: {count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    cv2.imshow("Collect Data", img)

    key = cv2.waitKey(1)

    if key == ord("s"):
        file_name = f"{save_path}/{count}.jpg"
        cv2.imwrite(file_name, img)
        count += 1

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()