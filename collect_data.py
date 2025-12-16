import cv2
import numpy as np
import pandas as pd
import os

# Define the colors and their key bindings
COLORS = {
    'r': 'Red',
    'o': 'Orange',
    'y': 'Yellow',
    'g': 'Green',
    'b': 'Blue',
    'i': 'Indigo',
    'v': 'Violet',
    'p': 'Purple',
    'w': 'White',
    'k': 'Black'
}

DATA_FILE = 'colors.csv'

def save_data(b, g, r, label):
    # Check if file exists to write header
    file_exists = os.path.isfile(DATA_FILE)
    
    with open(DATA_FILE, 'a') as f:
        if not file_exists:
            f.write('B,G,R,Label\n')
        f.write(f'{b},{g},{r},{label}\n')
    print(f"Saved: B={b}, G={g}, R={r} -> {label}")

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("--- Color Data Collector ---")
    print("Point the camera center at a color and press the corresponding key:")
    for k, v in COLORS.items():
        print(f"'{k}' -> {v}")
    print("'q' -> Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Define center point
        cx, cy = width // 2, height // 2
        
        # Get color at center
        # Note: OpenCV uses BGR ordering
        pixel_center = frame[cy, cx]
        b, g, r = int(pixel_center[0]), int(pixel_center[1]), int(pixel_center[2])
        
        # visuals
        cv2.circle(frame, (cx, cy), 10, (255, 255, 255), 2)
        cv2.putText(frame, "Target", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display current RGB
        cv2.putText(frame, f"BGR: {b},{g},{r}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Data Collection', frame)

        key = cv2.waitKey(1) & 0xFF
        char_key = chr(key) if key != 255 else ''

        if char_key == 'q':
            break
        elif char_key in COLORS:
            save_data(b, g, r, COLORS[char_key])

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
