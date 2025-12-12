import os
import cv2
import numpy as np
import random

def resize_image(img, size=(128,128)):
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def random_augmentation(img):
    types = random.choice(['rotate', 'flip', 'crop', 'blur', 'brightness'])
    
    if types == "rotate":
        direction = random.choice(["clockwise", "counterclockwise"])
        if direction == "clockwise":
            new_image = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        else:
            new_image = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    elif types == "flip":
        direction = random.choice(['horizontal', 'vertical'])
        if direction == 'horizontal':
            new_image = cv2.flip(img, 1)
        else:
            new_image = cv2.flip(img, 0)

    elif types == "crop":
        h, w = img.shape[:2]  
        ratio = random.uniform(0.7, 0.9)
        new_h, new_w = int(h * ratio), int(w * ratio)
        start_x = random.randint(0, w - new_w)
        start_y = random.randint(0, h - new_h)
        new_image = img[start_y:start_y + new_h, start_x:start_x + new_w]

    elif types == "blur":
        k = random.choice([3, 5, 7])
        new_image = cv2.GaussianBlur(img, (k, k), 0)

    elif types == "brightness":
        alpha = random.uniform(1.0, 1.5)  
        beta = random.randint(10, 50)     
        new_image = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
    return new_image

basePath = r"Data/dataset"
augmentedPath = "Data/augmented_data"
materialClasses = os.listdir(basePath)
os.makedirs(augmentedPath, exist_ok=True)
target_count = 500  

for material_class in materialClasses:
    src_folder = os.path.join(basePath, material_class)
    dst_folder = os.path.join(augmentedPath, material_class)
    os.makedirs(dst_folder, exist_ok=True)
    images = [img for img in os.listdir(src_folder) if img.lower().endswith((".jpg", ".jpeg", ".png"))]
    current_count = len(images)
    
    for img_name in images:
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print("Skipping empty image:", img_path)
            continue
        img = resize_image(img)
        cv2.imwrite(os.path.join(dst_folder, img_name), img)
    
    while current_count < target_count:
        img_name = random.choice(images)
        img_path = os.path.join(src_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        new_image = random_augmentation(img)
        new_image = resize_image(new_image) 
        new_name = f"{material_class}_{current_count}.jpg"
        cv2.imwrite(os.path.join(dst_folder, new_name), new_image)
        current_count += 1