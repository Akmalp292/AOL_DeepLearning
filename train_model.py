import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. Load Data
DATA_FILE = 'colors.csv'
if not os.path.exists(DATA_FILE):
    print(f"Error: {DATA_FILE} not found. Please run collect_data.py first.")
    exit()

df = pd.read_csv(DATA_FILE)

# 2. Preprocessing
X = df[['B', 'G', 'R']].values.astype(float)
y = df['Label'].values

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features (0-255 -> 0-1)
X = X / 255.0

# One-hot encode targets
y_categorical = tf.keras.utils.to_categorical(y_encoded)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# 3. Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 4. Train
print("Training model...")
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1, validation_data=(X_test, y_test))

# 5. Save
model.save('color_model.h5')
np.save('label_encoder.npy', le.classes_)
print("Model saved to 'color_model.h5'")
print("Label encoder saved to 'label_encoder.npy'")
