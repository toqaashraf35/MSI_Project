import os
import numpy as np
import joblib
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
THRESHOLD = 0.6 

def predict(dataFilePath, bestModelPath):
    cnn_model = load_model("cnn_feature_extractor.h5")  
    label_encoder = joblib.load("label_encoder.pkl")
    model = joblib.load(bestModelPath) 
    predictions = []
    for imgName in os.listdir(dataFilePath):
        imgPath = os.path.join(dataFilePath, imgName)
        if not imgName.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img = cv2.imread(imgPath)
        if img is None:
            continue  
        img_resized = cv2.resize(img, IMG_SIZE)
        x = img_resized.astype("float32")
        x = preprocess_input(x)
        x = np.expand_dims(x, axis=0)
        features = cnn_model.predict(x, verbose=0)
        probs = model.predict_proba(features)[0]
        max_idx = np.argmax(probs)
        confidence = probs[max_idx]
        if confidence < THRESHOLD:
            predictions.append("Unknown")
        else:
            label = label_encoder.inverse_transform([max_idx])[0]
            predictions.append(label)

    return predictions