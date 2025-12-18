import cv2
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time
import os

cnn_model = load_model("cnn_feature_extractor.h5")
svm_model = joblib.load("svm_model.pkl")
knn_model = joblib.load("knn_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
IMG_SIZE = (224, 224)
THRESHOLD = 0.6

def extract_features(frame):
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype("float32")
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = cnn_model.predict(img, verbose=0)
    return features

def predict_material(frame, model):
    features = extract_features(frame)
    probs = model.predict_proba(features)[0]
    max_idx = np.argmax(probs)
    acuracy = probs[max_idx]
    if acuracy < THRESHOLD:
        return "Unknown", acuracy
    label = label_encoder.inverse_transform([max_idx])[0]
    return label, acuracy

print("Select the model:")
print("1. SVM")
print("2. kNN")
choice = input("Enter 1 or 2: ")

if choice == "1":
    model = svm_model
    print("SVM model")
elif choice == "2":
    model = knn_model
    print("kNN model")
else:
    print("Invalid choice")
    model = svm_model

while True:
    print("\nSelect mode:")
    print("1. Live Camera")
    print("2. Upload Image")
    mode = input("Enter 1 or 2: ")

    if mode == "1":
        DURATION = 60  
        cap = cv2.VideoCapture(0)
        startTime = time.time()
        frameCount = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frameCount += 1
            label, conf = predict_material(frame, model)
            fps = frameCount / (time.time() - startTime + 1e-6)

            text = f"{label} ({conf:.2f}) FPS: {fps:.1f}"
            cv2.putText(frame, text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (0, 255, 0), 2)
            cv2.imshow("Material Stream Identification", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if time.time() - startTime > DURATION:
                break
        cap.release()
        cv2.destroyAllWindows()

    elif mode == "2":
        imgPath = input("Enter path of image: ")
        if not os.path.exists(imgPath):
            print("Image not found")
            continue
        img = cv2.imread(imgPath)
        label, conf = predict_material(img, model)
        print(f"Prediction: {label} (acuracy: {conf:.2f})")
        cv2.putText(img, f"{label} ({conf:.2f})", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Uploaded Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("Invalid choice")
        break