import cv2
import numpy as np
import tensorflow as tf
import os

# Load model and label encoder
if not os.path.exists('color_model.h5') or not os.path.exists('label_encoder.npy'):
    print("Error: Model files not found. Please run train_model.py first.")
    exit()

model = tf.keras.models.load_model('color_model.h5')
class_names = np.load('label_encoder.npy', allow_pickle=True)

def get_prediction(b, g, r):
    # Normalize
    input_data = np.array([[b, g, r]], dtype=float) / 255.0
    prediction = model.predict(input_data, verbose=0)
    idx = np.argmax(prediction)
    return class_names[idx], prediction[0][idx]

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("--- Real-time Color Detector ---")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        # Define center ROI (50x50 box)
        box_size = 50
        x1 = width // 2 - box_size // 2
        y1 = height // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        
        # Get ROI
        roi = frame[y1:y2, x1:x2]
        
        # Calculate average color in ROI
        avg_color_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_row, axis=0)
        b, g, r = avg_color[0], avg_color[1], avg_color[2]
        
        # Predict
        label, confidence = get_prediction(b, g, r)

        # Visuals
        # Draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        
        # Display BGR and Prediction
        info_text = f"RGB: ({int(r)},{int(g)},{int(b)}) | Color: {label} ({confidence:.2f})"
        
        # Background for text
        cv2.rectangle(frame, (5, 5), (600, 60), (0, 0, 0), -1)
        cv2.putText(frame, info_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Real-time Color Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
