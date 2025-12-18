import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler

def color_histogram_feature(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0,1,2],
        None,
        [16,16,8],
        [0,180,0,256,0,256]
    )
    cv2.normalize(hist, hist)
    return hist.flatten()

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


basePath = r"Downloads/ml_project/data/augmented_data"
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
        class_to_id = {name: idx for idx, name in enumerate(materialClasses)}
        y.append(class_to_id[material_class])

X = np.array(X)
y = np.array(y)

print("Feature extraction complete!")
print("X shape:", X.shape)
print("y shape:", y.shape)




scaler = StandardScaler()
X = scaler.fit_transform(X)
