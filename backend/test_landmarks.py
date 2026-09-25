import cv2

from app.services.hand_detection import HandDetector


detector = HandDetector()

cap = cv2.VideoCapture(0)

print("Show one or both hands.")
print("Press Q to quit.")

while cap.isOpened():

    success, frame = cap.read()

    if not success:
        print("Could not access camera.")
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = detector.detect(rgb_frame)

    hands = detector.extract_landmarks(results)

    left = hands["left"]
    right = hands["right"]

    features = detector.create_feature_vector(hands)

    print(
    f"Left: {'YES' if left else 'NO'} | "
    f"Right: {'YES' if right else 'NO'} | "
    f"Features: {len(features)}"
)

    cv2.imshow(
        "Two Hand Feature Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

detector.close()

print("\nTest completed.")