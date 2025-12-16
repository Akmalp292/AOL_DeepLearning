import tensorflow as tf
import os

def inspect(model_path):
    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        return

    try:
        model = tf.keras.models.load_model(model_path)
        print(f"--- Model: {model_path} ---")
        
        # Input shape
        if hasattr(model, 'input_shape'):
             print(f"Input Shape: {model.input_shape}")
        
        # Output shape
        if hasattr(model, 'output_shape'):
            print(f"Output Shape: {model.output_shape}")
            
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")

inspect('final_model.h5')
inspect('model_best.keras')
