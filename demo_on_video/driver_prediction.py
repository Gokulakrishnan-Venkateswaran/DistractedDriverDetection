#driver_prediction.py
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json

model = load_model('../models/model.h5')
with open("../class_name_map.json") as f:
    class_map = json.load(f)

cap = cv2.VideoCapture('input_video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    resized = cv2.resize(frame, (150, 150))
    img_array = np.expand_dims(resized / 255.0, axis=0)
    pred = model.predict(img_array)
    label = class_map[str(np.argmax(pred))]
    cv2.putText(frame, f'Detected: {label}', (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()