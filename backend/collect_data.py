import cv2
import os
import time
import numpy as np

from app.services.hand_detection import HandDetector


# ============================================================
# CONFIGURATION
# ============================================================

SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "I_LOVE_YOU",
]

# Number of frames in one sample
SEQUENCE_LENGTH = 30

# Number of samples to collect for each sign
TOTAL_SAMPLES = 50

# Dataset folder
DATASET_DIR = "dataset"

# Camera window name
WINDOW_NAME = "Sign Language Dataset Collection"


# ============================================================
# HAND DETECTOR
# ============================================================

detector = HandDetector()


# ============================================================
# GET NEXT SAMPLE NUMBER
# ============================================================

def get_next_sample_number(sign_dir):
    """
    Find the next available sample number.

    Example:
        sample_001.npy
        sample_002.npy

    Next number = 3
    """

    existing_files = [
        file
        for file in os.listdir(sign_dir)
        if file.endswith(".npy")
    ]

    if not existing_files:
        return 1

    numbers = []

    for file in existing_files:

        try:

            number = int(
                file.replace("sample_", "")
                    .replace(".npy", "")
            )

            numbers.append(number)

        except ValueError:

            continue

    if not numbers:
        return 1

    return max(numbers) + 1


# ============================================================
# SHOW MESSAGE ON CAMERA
# ============================================================

def show_message(frame, message, y=60, scale=0.9):

    cv2.putText(
        frame,
        message,
        (30, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 255),
        2,
        cv2.LINE_AA
    )


# ============================================================
# COLLECT ONE SEQUENCE
# ============================================================

def collect_sequence(cap, sign, sample_number):

    print()
    print("-" * 55)
    print(
        f"Preparing sample "
        f"{sample_number}/{TOTAL_SAMPLES}"
    )
    print("-" * 55)

    # --------------------------------------------------------
    # GET READY
    # --------------------------------------------------------

    ready_start = time.time()

    while time.time() - ready_start < 1.5:

        success, frame = cap.read()

        if not success:

            print("Could not read camera frame.")

            return None

        frame = cv2.flip(frame, 1)

        show_message(
            frame,
            f"Get ready... "
            f"Sample {sample_number}/{TOTAL_SAMPLES}",
            60,
            0.8
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return None

    # --------------------------------------------------------
    # COUNTDOWN
    # --------------------------------------------------------

    for countdown in range(3, 0, -1):

        countdown_start = time.time()

        while time.time() - countdown_start < 1:

            success, frame = cap.read()

            if not success:

                print("Could not read camera frame.")

                return None

            frame = cv2.flip(frame, 1)

            show_message(
                frame,
                f"Starting in {countdown}",
                60,
                1.2
            )

            show_message(
                frame,
                f"Sample {sample_number}/{TOTAL_SAMPLES}",
                110,
                0.8
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                return None

    print("START!")

    # --------------------------------------------------------
    # RECORD 30 FRAMES
    # --------------------------------------------------------

    sequence = []

    previous_features = None

    while len(sequence) < SEQUENCE_LENGTH:

        success, frame = cap.read()

        if not success:

            print("Could not read camera frame.")

            return None

        # Mirror camera
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Detect hands
        results = detector.detect(
            rgb_frame
        )

        # Extract left/right landmarks
        hands = detector.extract_landmarks(
            results
        )

        # Create fixed 126-feature vector
        features = detector.create_feature_vector(
            hands
        )

        # ----------------------------------------------------
        # CHECK HAND DETECTION
        # ----------------------------------------------------

        hand_detected = (
            hands["left"] is not None
            or
            hands["right"] is not None
        )

        if hand_detected:

            # Store latest valid frame
            previous_features = features

        elif previous_features is not None:

            # If tracking temporarily fails,
            # use previous valid frame.
            features = previous_features

        else:

            # No hand detected yet
            show_message(
                frame,
                "Show your hand",
                55,
                0.9
            )

            show_message(
                frame,
                f"Sample {sample_number}/{TOTAL_SAMPLES}",
                100,
                0.8
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                return None

            continue

        # ----------------------------------------------------
        # ADD FRAME
        # ----------------------------------------------------

        sequence.append(features)

        # ----------------------------------------------------
        # DISPLAY PROGRESS
        # ----------------------------------------------------

        show_message(
            frame,
            f"Recording: "
            f"{len(sequence)}/{SEQUENCE_LENGTH}",
            50,
            0.9
        )

        show_message(
            frame,
            f"Sample: "
            f"{sample_number}/{TOTAL_SAMPLES}",
            95,
            0.8
        )

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            return None

    # --------------------------------------------------------
    # CONVERT TO NUMPY
    # --------------------------------------------------------

    sequence = np.array(
        sequence,
        dtype=np.float32
    )

    # Safety check
    if sequence.shape != (
        SEQUENCE_LENGTH,
        126
    ):

        print(
            "ERROR: Unexpected sequence shape:",
            sequence.shape
        )

        return None

    return sequence


# ============================================================
# COLLECT 50 SAMPLES FOR ONE SIGN
# ============================================================

def collect_sign(sign):

    # --------------------------------------------------------
    # CREATE SIGN DIRECTORY
    # --------------------------------------------------------

    sign_dir = os.path.join(
        DATASET_DIR,
        sign
    )

    os.makedirs(
        sign_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # FIND STARTING SAMPLE NUMBER
    # --------------------------------------------------------

    next_sample = get_next_sample_number(
        sign_dir
    )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print()
        print("ERROR: Could not access camera.")

        return

    # --------------------------------------------------------
    # CAMERA WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    # --------------------------------------------------------
    # START MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print(
        f"COLLECTING DATA FOR: {sign}"
    )
    print(
        f"TARGET: {TOTAL_SAMPLES} SAMPLES"
    )
    print("=" * 60)

    print()
    print("Camera opened successfully.")
    print("Perform the sign naturally.")
    print("Press Q to stop at any time.")
    print()

    # --------------------------------------------------------
    # COLLECTION LOOP
    # --------------------------------------------------------

    samples_collected = 0

    stopped_by_user = False

    while samples_collected < TOTAL_SAMPLES:

        current_sample = (
            next_sample + samples_collected
        )

        # ----------------------------------------------------
        # COLLECT ONE SEQUENCE
        # ----------------------------------------------------

        sequence = collect_sequence(
            cap,
            sign,
            samples_collected + 1
        )

        # ----------------------------------------------------
        # STOP IF Q WAS PRESSED
        # ----------------------------------------------------

        if sequence is None:

            stopped_by_user = True

            print()
            print("Collection stopped.")

            break

        # ----------------------------------------------------
        # FILE NAME
        # ----------------------------------------------------

        filename = (
            f"sample_{current_sample:03d}.npy"
        )

        filepath = os.path.join(
            sign_dir,
            filename
        )

        # ----------------------------------------------------
        # SAVE SAMPLE
        # ----------------------------------------------------

        np.save(
            filepath,
            sequence
        )

        samples_collected += 1

        print(
            f"Saved: {filepath}"
        )

        print(
            f"Progress: "
            f"{samples_collected}/{TOTAL_SAMPLES}"
        )

        # ----------------------------------------------------
        # WAIT BEFORE NEXT SAMPLE
        # ----------------------------------------------------

        if samples_collected < TOTAL_SAMPLES:

            gap_start = time.time()

            while time.time() - gap_start < 1.5:

                success, frame = cap.read()

                if not success:

                    break

                frame = cv2.flip(
                    frame,
                    1
                )

                show_message(
                    frame,
                    "Get ready for next sample...",
                    60,
                    0.8
                )

                show_message(
                    frame,
                    f"Progress: "
                    f"{samples_collected}/{TOTAL_SAMPLES}",
                    105,
                    0.8
                )

                cv2.imshow(
                    WINDOW_NAME,
                    frame
                )

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):

                    stopped_by_user = True

                    break

            if stopped_by_user:

                break

    # --------------------------------------------------------
    # CLOSE CAMERA
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    # --------------------------------------------------------
    # FINAL MESSAGE
    # --------------------------------------------------------

    print()
    print("=" * 60)

    if samples_collected == TOTAL_SAMPLES:

        print(
            f"SUCCESS: {sign}"
        )

        print(
            f"All {TOTAL_SAMPLES} samples collected!"
        )

    else:

        print(
            f"STOPPED: {sign}"
        )

        print(
            f"Samples collected: "
            f"{samples_collected}/{TOTAL_SAMPLES}"
        )

    print("=" * 60)
    print()


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print()
    print("=" * 45)
    print("SIGN LANGUAGE DATASET COLLECTOR")
    print("=" * 45)

    print()
    print("Available signs:")

    for index, sign in enumerate(
        SIGNS,
        start=1
    ):

        print(
            f"{index}. {sign}"
        )

    print()

    choice = input(
        "Choose sign number: "
    )

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    try:

        choice = int(choice)

    except ValueError:

        print()
        print(
            "Invalid input. "
            "Please enter a number."
        )

        return

    if choice < 1 or choice > len(SIGNS):

        print()
        print(
            "Invalid choice."
        )

        return

    # --------------------------------------------------------
    # SELECT SIGN
    # --------------------------------------------------------

    sign = SIGNS[
        choice - 1
    ]

    # --------------------------------------------------------
    # START COLLECTION
    # --------------------------------------------------------

    collect_sign(sign)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print("Program interrupted.")

    finally:

        detector.close()

        cv2.destroyAllWindows()