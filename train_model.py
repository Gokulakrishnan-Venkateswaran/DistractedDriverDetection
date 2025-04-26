# train_model.py
import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Prepare data
train_dir = 'data/train'
img_size = 150
batch_size = 32

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = datagen.flow_from_directory(train_dir,
                                         target_size=(img_size, img_size),
                                         batch_size=batch_size,
                                         class_mode='categorical',
                                         subset='training')

val_data = datagen.flow_from_directory(train_dir,
                                       target_size=(img_size, img_size),
                                       batch_size=batch_size,
                                       class_mode='categorical',
                                       subset='validation')

# CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_data, validation_data=val_data, epochs=10)

model.save('models/model.h5')
