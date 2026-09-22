import cv2
import mediapipe as mp

from app.services.hand_detection import HandDetector


detector = HandDetector()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Could not access camera")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = detector.detect(rgb_frame)

    hands = detector.extract_landmarks(results)

    if hands:
        print(f"Hands detected: {len(hands)}")
        print(f"Landmarks in first hand: {len(hands[0])}")

        normalized = detector.normalize_landmarks(hands[0])

        print(f"Normalized features: {len(normalized)}")
        print(f"First 6 values: {normalized[:6]}")
        print("-" * 40)

    cv2.imshow("Landmark Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
detector.close()
cv2.destroyAllWindows()