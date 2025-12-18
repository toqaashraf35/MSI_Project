import os
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import joblib
from google.colab import files

dataPath = "/content/data"
imgSize = (224, 224)
baseModel = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')

features = []
labels = []

classNames = sorted(os.listdir(dataPath)) 

for materialClass in classNames:
    classPath = os.path.join(dataPath, materialClass)
    if not os.path.isdir(classPath):
        continue
    for imgName in os.listdir(classPath):
        if not imgName.lower().endswith(('.jpg','.png','.jpeg')):
            continue
        imgPath = os.path.join(classPath, imgName)
        try:
            img = load_img(imgPath, target_size=imgSize)
        except:
            continue
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x) 
        featureVector = baseModel.predict(x, verbose=0)
        features.append(featureVector.flatten())
        labels.append(materialClass)
features = np.array(features)
labels = np.array(labels)

np.save('features.npy', features)
np.save('labels.npy', labels)
files.download('features.npy')
files.download('labels.npy')
baseModel.save("cnn_feature_extractor.h5")
baseModel.save("cnn_feature_extractor.keras")

