import csv
import os
import cv2

from app.services.hand_detection import HandDetector


# Signs we want to collect
SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "HELP",
]

DATASET_DIR = "dataset"

detector = HandDetector()


def collect_sign(sign):
    """
    Collect landmark samples for one sign.
    """

    sign_dir = os.path.join(DATASET_DIR, sign)
    os.makedirs(sign_dir, exist_ok=True)

    csv_file = os.path.join(sign_dir, "landmarks.csv")

    cap = cv2.VideoCapture(0)

    print()
    print("=" * 50)
    print(f"Collecting data for: {sign}")
    print("Show the sign to the camera.")
    print("Press SPACE to capture a sample.")
    print("Press Q to stop.")
    print("=" * 50)

    with open(csv_file, "a", newline="") as file:

        writer = csv.writer(file)

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

            # Display information
            cv2.putText(
                frame,
                f"Sign: {sign}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "SPACE = Capture | Q = Quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if hands:
                cv2.putText(
                    frame,
                    "HAND DETECTED",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Sign Language Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            # Capture sample
            if key == ord(" "):

                if hands:

                    normalized = detector.normalize_landmarks(
                        hands[0]
                    )

                    writer.writerow(normalized)

                    print(
                        f"Sample saved for {sign} "
                        f"({len(normalized)} features)"
                    )

                else:
                    print("No hand detected. Try again.")

            # Quit
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def main():

    print("\nSign Language Dataset Collector")
    print("--------------------------------")

    for index, sign in enumerate(SIGNS, start=1):
        print(f"{index}. {sign}")

    print("0. Exit")

    choice = input("\nEnter sign number: ")

    if choice == "0":
        return

    try:
        sign_index = int(choice) - 1
        sign = SIGNS[sign_index]

    except (ValueError, IndexError):
        print("Invalid choice.")
        return

    collect_sign(sign)

    detector.close()


if __name__ == "__main__":
    main()