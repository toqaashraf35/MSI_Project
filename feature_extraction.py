import os
import cv2
import numpy as np
from skimage.feature import hog

def color_histogram_feature(img, bins = 32):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([img_hsv], [0], None, [bins], [0, 180])  
    s = cv2.calcHist([img_hsv], [1], None, [bins], [0, 256])  
    v = cv2.calcHist([img_hsv], [2], None, [bins], [0, 256]) 
    h = h / (h.sum() + 1e-6)
    s = s / (s.sum() + 1e-6)
    v = v / (v.sum() + 1e-6)
    return np.concatenate([h.flatten(), s.flatten(), v.flatten()])

def HOG_feature(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        img_gray,
        orientations=9,     
        pixels_per_cell=(8,8),      
        cells_per_block=(2,2),      
        block_norm='L2-Hys',       
        transform_sqrt=True,        
        feature_vector=True         
    )
    return hog_features


basePath = r"Data/augmented_data"
materialClasses = os.listdir(basePath)

X = []
y = []
for material_class in materialClasses:
    folderPath = os.path.join(basePath, material_class)
    images = os.listdir(folderPath)

    for img_name in images:
        img_path = os.path.join(folderPath, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue
        color_features = color_histogram_feature(img)
        hog_features = HOG_feature(img)
        combined_features = np.hstack([hog_features, color_features])
        X.append(combined_features)
        y.append(material_class)

X = np.array(X)
y = np.array(y)

np.save("Data/X_features.npy", X)
np.save("Data/y_labels.npy", y)
print("X shape:", X.shape)
print("y shape:", y.shape)

