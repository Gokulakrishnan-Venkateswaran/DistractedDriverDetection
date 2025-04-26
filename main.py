# main.py
import os
import numpy as np
import cv2
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Constants
IMAGE_SIZE = 64
DATA_PATH = "train/"

def load_data():
    images, labels = [], []
    for label in range(10):
        folder = os.path.join(DATA_PATH, f"c{label}")
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            images.append(img)
            labels.append(label)
    return np.array(images), to_categorical(labels, 10)

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train
X, y = load_data()
X = X / 255.0
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_model()
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)

# Save model
model.save("driver_model.h5")