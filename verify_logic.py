import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import cv2

def verify():
    print("Beginning Verification...")
    
    # 1. Check Files
    required_files = ['final_model.h5', 'color_model.h5', 'label_encoder.npy']
    for f in required_files:
        if not os.path.exists(f):
            print(f"FAILED: Missing {f}")
            return

    # 2. Load Models
    try:
        print("Loading Ishihara Model...")
        ishihara_model = tf.keras.models.load_model('final_model.h5')
        print("Ishihara Model Loaded.")
        
        print("Loading Color Model...")
        color_model = tf.keras.models.load_model('color_model.h5')
        print("Color Model Loaded.")
        
        class_names = np.load('label_encoder.npy', allow_pickle=True)
        print(f"Classes: {class_names}")
    except Exception as e:
        print(f"FAILED to load models: {e}")
        return

    # 3. Dummy Inference (Ishihara)
    try:
        print("Testing Ishihara Inference...")
        # Create dummy image
        img = Image.new('RGB', (300, 300), color = 'red')
        
        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        pred = ishihara_model.predict(img_array, verbose=0)
        print(f"Ishihara Prediction Shape: {pred.shape}")
    except Exception as e:
        print(f"FAILED Ishihara Inference: {e}")
        return

    # 4. Dummy Inference (Color)
    try:
        print("Testing Color Inference...")
        # Create dummy numpy image (like cv2/PIL array)
        img_array_full = np.array(img) # 300x300x3
        
        # ROI Logic
        h, w, _ = img_array_full.shape
        box_size = 50
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        roi = img_array_full[y1:y1+box_size, x1:x1+box_size]
        
        avg_color_row = np.average(roi, axis=0)
        avg_color = np.average(avg_color_row, axis=0)
        b, g, r = avg_color[0], avg_color[1], avg_color[2] # If RGB
        
        # App logic assumes R, G, B order from PIL, but earlier I saw main.py uses BGR from OpenCV.
        # app.py reads PIL -> RGB.
        # main.py reads OpenCV -> BGR.
        # train_model.py trained on colors.csv.
        # collect_data.py collected B, G, R.
        # Check collect_data.py:
        # pixel_center = frame[cy, cx] (BGR)
        # b, g, r = ...
        # save_data(b, g, r, ...)
        # So Model expects [B, G, R] inputs.
        
        # app.py:
        # image = Image.open(...) -> RGB
        # avg_color = ... (R, G, B) order because input is RGB
        # r, g, b = avg_color[0], avg_color[1], avg_color[2]
        # input_data = [[b, g, r]]
        # This matches!
        
        input_data = np.array([[b, g, r]], dtype=float) / 255.0
        
        pred = color_model.predict(input_data, verbose=0)
        print(f"Color Prediction Shape: {pred.shape}")
    except Exception as e:
        print(f"FAILED Color Inference: {e}")
        return

    print("VERIFICATION PASSED")

if __name__ == "__main__":
    verify()
