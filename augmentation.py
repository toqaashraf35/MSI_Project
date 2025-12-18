import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img

dataPath = "/content/data" 
targetCount = 500
imgSize = (224, 224)

dataGenerator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

def augmentation(classPath):
    images = [img for img in os.listdir(classPath)
              if img.lower().endswith(('.jpg', '.png', '.jpeg'))]
    i = 0

    while len(images) < targetCount:
        imgName = images[i % len(images)]
        imgPath = os.path.join(classPath, imgName)
        try:
            img = load_img(imgPath, target_size=imgSize)
        except Exception as e:
            images.remove(imgName)
            i += 1
            continue
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        for batch in dataGenerator.flow(x, batchSize=1):
            newImgName = f"aug_{len(images)}.jpg"
            savePath = os.path.join(classPath, newImgName)
            save_img(savePath, batch[0])
            images.append(newImgName)
            break
        i += 1

if __name__ == "__main__":
    for materialClass in os.listdir(dataPath):
        classPath = os.path.join(dataPath, materialClass)
        if os.path.isdir(classPath):
            augmentation(classPath)
    print("All classes are done")