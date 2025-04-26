# predictionOnImage.py

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

model = load_model("models/model.h5")

with open("class_name_map.json") as f:
    class_map = json.load(f)

img = cv2.imread("test_image.jpg")
img_resized = cv2.resize(img, (150, 150))
img_array = np.expand_dims(img_resized / 255.0, axis=0)

prediction = model.predict(img_array)
predicted_class = class_map[str(np.argmax(prediction))]

print(f"Predicted class: {predicted_class}")