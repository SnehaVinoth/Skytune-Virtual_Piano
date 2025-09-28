import cv2
import mediapipe as mp
import time
import csv
import numpy as np
import os

# Mediapipe helpers
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Notes
NOTE_LABELS = ["C", "D", "E", "F", "G", "A", "B"]

# CSV File Path
output_file = "Dataset/dataset.csv"
os.makedirs("Dataset", exist_ok=True)

# Write header if CSV doesn't exist
header = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")] + NOTE_LABELS
if os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

# Load keyboard overlay 
keyboard_overlay = cv2.imread("Dataset/keyboard.png", cv2.IMREAD_UNCHANGED)

def overlay_image(background, overlay, x=0, y=0):
    bh, bw = background.shape[:2]
    oh, ow = overlay.shape[:2]

    # Clip overlay if it's bigger than background
    if x + ow > bw:
        overlay = overlay[:, :bw-x]
        ow = overlay.shape[1]
    if y + oh > bh:
        overlay = overlay[:bh-y, :]
        oh = overlay.shape[0]

    # Ensure overlay has alpha channel
    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [overlay, np.ones((oh, ow, 1), dtype=overlay.dtype) * 255],
            axis=2,
        )

    # Separate BGR and alpha
    b,g,r,a = cv2.split(overlay)
    overlay_color = cv2.merge((b,g,r))
    mask = cv2.merge((a,a,a)).astype(float)/255

    background_part = background[y:y+oh, x:x+ow].astype(float)
    overlay_part = overlay_color.astype(float)

    blended = background_part * (1 - mask) + overlay_part * mask
    background[y:y+oh, x:x+ow] = blended.astype(np.uint8)
    return background

# Open webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(static_image_mode=False,
                    max_num_hands=1,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5) as hands:

    while True:
        i = input("Press Enter to start 5-second recording (or type q and Enter to quit): ")
        if i.lower() == 'q':
            print("Exiting...")
            break

        last_frame_landmarks = None
        start_time = time.time()
        print("Recording for 5 seconds")

        while time.time() - start_time < 5:
            ret, frame = cap.read()
            if not ret:
                continue

            # Flip for natural feel
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Overlay keyboard at bottom-center with original size
            y_offset = h - keyboard_overlay.shape[0]  # place at bottom
            x_offset = (w - keyboard_overlay.shape[1]) // 2  # center horizontally
            frame = overlay_image(frame, keyboard_overlay, x=x_offset, y=y_offset)


            # Convert BGR → RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmark_style = mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2)
                    connection_style = mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS, landmark_style, connection_style)
                    last_frame_landmarks = hand_landmarks  # keep last frame

            # Show webcam
            cv2.imshow("Virtual Piano Dataset", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped by user.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

        # After 5 seconds, ask for note label
        if last_frame_landmarks:
            while True:
                note_label = input(f"Type the note you just played ({'/'.join(NOTE_LABELS)}): ").upper()
                if note_label in NOTE_LABELS:
                    note_index = NOTE_LABELS.index(note_label)
                    break
                else:
                    print("Invalid note, try again.")

            # Flatten landmarks and create label row
            row = list(np.array([[lm.x, lm.y, lm.z] for lm in last_frame_landmarks.landmark]).flatten())
            label_row = [0]*7
            label_row[note_index] = 1
            row.extend(label_row)

            # Save to CSV
            with open(output_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)

            print(f"Saved note {note_label} to dataset!\n")

cap.release()
cv2.destroyAllWindows()
