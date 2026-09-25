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

        return self.hands.process(frame)

    def extract_landmarks(self, results):

        left_hand = None
        right_hand = None

        if not results.multi_hand_landmarks:
            return {
                "left": None,
                "right": None,
            }

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):

            hand_label = handedness.classification[0].label

            landmarks = []

            for landmark in hand_landmarks.landmark:

                landmarks.append([
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ])

            normalized = self.normalize_landmarks(
                landmarks
            )

            if hand_label == "Left":
                left_hand = normalized

            elif hand_label == "Right":
                right_hand = normalized

        return {
            "left": left_hand,
            "right": right_hand,
        }

    def normalize_landmarks(self, landmarks):

        if not landmarks:
            return []

        wrist = landmarks[0]

        normalized = []

        for landmark in landmarks:

            x = landmark[0] - wrist[0]
            y = landmark[1] - wrist[1]
            z = landmark[2] - wrist[2]

            normalized.extend([
                x,
                y,
                z,
            ])

        return normalized

    def create_feature_vector(self, hands):

        """
        Create a fixed 126-feature vector.

        Left hand  = 63 features
        Right hand = 63 features

        Missing hand = 63 zeros
        """

        left = hands["left"]
        right = hands["right"]

        if left is None:
            left = [0.0] * 63

        if right is None:
            right = [0.0] * 63

        return left + right

    def close(self):

        self.hands.close()