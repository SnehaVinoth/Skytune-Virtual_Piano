import cv2
import mediapipe as mp
import time
import csv

# Mediapipe helpers
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

NOTE_LABELS = ["C", "D", "E", "F", "G", "A", "B"]
KEY_TO_INDEX = {
    ord('c'): 0,
    ord('d'): 1,
    ord('e'): 2,
    ord('f'): 3,
    ord('g'): 4,
    ord('a'): 5,
    ord('b'): 6
}
# CSV File Path
output_file = "Dataset/dataset.csv"

header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + ["label"]

with open(output_file, mode="w", newline="") as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(header)

# Capture Video
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5) as hands:

    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Flip for natural feel
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Convert BGR → RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        # Back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Show webcam
        cv2.imshow("Virtual Piano Dataset", image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Recording stopped by user.")
            break

        if results.multi_hand_landmarks and key in KEY_TO_INDEX:
            note_index = KEY_TO_INDEX[key]
            for hand_landmarks in results.multi_hand_landmarks:
                try:
                    # Flatten 21 landmarks (x,y,z)
                    row = list(np.array([[lm.x, lm.y, lm.z] 
                                for lm in hand_landmarks.landmark]).flatten())

                    # Create label vector (all zeros except pressed note)
                    label_row = [0]*7
                    label_row[note_index] = 1
                    row.extend(label_row)

                    # Append to CSV
                    with open(CSV_PATH, mode='a', newline="") as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)

                    print(f"Saved frame for note: {NOTE_LABELS[note_index]}")

                except Exception as e:
                    print("Error saving row:", e)
                    pass
cap.release()
cv2.destroyAllWindows()



