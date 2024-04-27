from flask import Flask, render_template, request
import os
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
app = Flask(__name__)

model = load_model('medical_materials_model8.h5')
output_layer = model.output
train_subfolders = ['Aloevera','betel','castor','guava','hisbiscus','mango','neem','not','tulsi']

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
         return f"{predicted_class}"
    else:
        return "Not a medicinal plant"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        image_path = 'uploaded_image.jpg'
        file.save(image_path)
        original_image = cv2.imread('uploaded_image.jpg')

        hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])  
        upper_green = np.array([85, 255, 255])  

        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

        if green_mask.shape != original_image.shape[:2]:
           green_mask = cv2.resize(green_mask, (original_image.shape[1], original_image.shape[0]))

        result = cv2.bitwise_and(original_image, original_image, mask=green_mask)
        cv2.imwrite('green_leaves_image.jpg', result)
        image_path ='green_leaves_image.jpg'
        predicted_result = predict_plant(image_path, threshold=0.5)

        return predicted_result

if __name__ == '__main__':
    app.run(debug=True, port=8000, host='0.0.0.0')
