import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from sklearn.cluster import KMeans

# Set page config
st.set_page_config(page_title="Vision Project: Ishihara & Color", layout="wide")

# Load Models (Cached for performance)
@st.cache_resource
def load_models():
    # Load Ishihara Model
    try:
        ishihara_model = tf.keras.models.load_model('final_model.h5')
    except:
        ishihara_model = None
        st.error("Failed to load final_model.h5")

    # Load Color Model
    try:
        color_model = tf.keras.models.load_model('color_model.h5')
    except:
        color_model = None
        st.error("Failed to load color_model.h5")

    # Load Label Encoder
    try:
        class_names = np.load('label_encoder.npy', allow_pickle=True)
    except:
        class_names = None
        st.error("Failed to load label_encoder.npy")
        
    return ishihara_model, color_model, class_names

ishihara_model, color_model, class_names = load_models()

def predict_ishihara(image, model):
    if model is None:
        return "Model Error"
    
    # Preprocess: Resize to 224x224, Normalize
    # Image is expected to be RGB
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0 # Normalize 0-1
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    return f"{predicted_class} ({confidence*100:.1f}%)"

def get_dominant_colors(image, k=10):
    # Resize for speed
    img_small = image.resize((100, 100))
    img_array = np.array(img_small)
    
    # Reshape to list of pixels
    pixels = img_array.reshape(-1, 3)
    
    # K-Means
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_
    
    # Calculate dominance (count labels) - useful for sorting internal processing
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    
    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    sorted_colors = colors[sorted_indices]
    
    return sorted_colors

def predict_colors_multi(image, model, class_names):
    if model is None or class_names is None:
        return []
        
    dominant_rgbs = get_dominant_colors(image, k=10)
    results = []
    seen_labels = set()
    
    for i, rgb in enumerate(dominant_rgbs):
        r, g, b = rgb[0], rgb[1], rgb[2]
        
        # Predict label
        input_data = np.array([[b, g, r]], dtype=float) / 255.0 # BGR order for model
        prediction = model.predict(input_data, verbose=0)
        idx = np.argmax(prediction)
        label = class_names[idx]
        confidence = prediction[0][idx]
        
        # Add to results if we haven't seen this label yet, OR if we want to show all shades.
        # User asked for "All the colors in image". 
        # Strategy: Show all detected unique color classes.
        
        if label not in seen_labels:
             results.append({
                'label': label,
                'rgb': (int(r), int(g), int(b)),
                'conf': confidence
            })
             seen_labels.add(label)
        
    return results

# --- UI ---
st.title("👁️ Vision System: Ishihara & Color Detection")
st.write("Detects Ishihara blind test numbers (0-9) and RGB colors simultaneously.")

# Input
input_source = st.radio("Select Input Source:", ("Camera", "Upload Image"))

image = None

if input_source == "Camera":
    img_file = st.camera_input("Take a picture")
    if img_file is not None:
        image = Image.open(img_file)
else:
    img_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    if img_file is not None:
        image = Image.open(img_file)

if image is not None:
    # Display Image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Captured Image", use_column_width=True)

    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing..."):
            # 1. Ishihara Detection
            ishihara_result = predict_ishihara(image, ishihara_model)
            st.metric("Ishihara Number", ishihara_result)
            
            st.divider()
            
            # 2. Multi-Color Detection
            st.divider()
            
            # Get Results
            color_results = predict_colors_multi(image, color_model, class_names)
            
            # Extract unique names
            detected_names = [res['label'] for res in color_results]
            unique_names_str = ", ".join(detected_names)
            
            st.write(f"### Detected Colors: {len(detected_names)}")
            st.success(f"**Found:** {unique_names_str}")
            
            # Show Palette
            st.write("#### Color Samples Extracted")
            
            # Display logic
            cols = st.columns(len(color_results)) if len(color_results) > 0 else []
            
            for idx, res in enumerate(color_results):
                c_label = res['label']
                c_rgb = res['rgb']
                hex_color = f"#{c_rgb[0]:02x}{c_rgb[1]:02x}{c_rgb[2]:02x}"
                
                with cols[idx]:
                    st.color_picker(f"{c_label}", hex_color, disabled=True, label_visibility="visible", key=f"c_{idx}_{hex_color}")
            
