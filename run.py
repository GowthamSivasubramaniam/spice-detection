from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import cv2

train_path = 'Dataset/Train'
model = load_model('spices.h5')
output_layer = model.output
train_subfolders = [folder for folder in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, folder))]

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)
    return img

def predict_plant(image_path, threshold=0.5):
    preprocessed_image = preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    print(predictions)
    class_labels = train_subfolders
    
    max_confidence = np.max(predictions)
    predicted_class = class_labels[np.argmax(predictions)]
    
    if max_confidence >= threshold:
        if(predicted_class=='Not'):
         return f"Not a Medicinal Plant"
        else:
         return f"The predicted Spice is: {predicted_class} (Confidence: {max_confidence})"
    else:
        return "Not a medicinal spice"
image = cv2.imread('WhatsApp Image 2023-09-27 at 01.41.32.jpg')

lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)

mask = cv2.inRange(image, lower_white, upper_white)


image[mask > 0] = [0, 0, 0]  

cv2.imwrite('image_with_black_background.jpg', image)
image_path='image_with_black_background.jpg'
predicted_result = predict_plant(image_path, threshold=0.7)  
print(predicted_result)
