import mediapipe as mp


class HandDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def detect(self, frame):
        """
        Detect hands in an RGB image.
        """
        return self.hands.process(frame)

    def extract_landmarks(self, results):
        """
        Extract x, y, z coordinates from detected hands.
        """

        all_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                landmarks = []

                for landmark in hand_landmarks.landmark:
                    landmarks.append([
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ])

                all_hands.append(landmarks)

        return all_hands

    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to the wrist.
        Returns 63 values:
        21 landmarks × 3 coordinates.
        """

        if not landmarks:
            return []

        wrist = landmarks[0]

        normalized = []

        for landmark in landmarks:
            x = landmark[0] - wrist[0]
            y = landmark[1] - wrist[1]
            z = landmark[2] - wrist[2]

            normalized.extend([x, y, z])

        return normalized

    def close(self):
        self.hands.close()