import cv2
# import mediapipe as mp
import mediapipe as mp
print(dir(mp))

class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.mp_draw = mp.solutions.drawing_utils

    def detect_hands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        landmarks_list = []

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)

                for lm in handLms.landmark:
                    landmarks_list.append([lm.x, lm.y])

        return img, landmarks_list


# TEST RUN
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        success, img = cap.read()
        img, landmarks = detector.detect_hands(img)

        if landmarks:
            print("Landmarks:", len(landmarks))  # should print 21

        cv2.imshow("Hand Detection", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
