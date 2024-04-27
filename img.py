import os
import cv2
import numpy as np

input_folder = 'raw materials/turmeric'  
output_folder = 'Dataset/Train/Turmeric'  

lower_white = np.array([200, 200, 200], dtype=np.uint8)
upper_white = np.array([255, 255, 255], dtype=np.uint8)


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image = cv2.imread(os.path.join(input_folder, filename))

        mask = cv2.inRange(image, lower_white, upper_white)

        image[mask == 255] = [0, 0, 0]  

       
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, image)

        print(f"Processed: {filename}")

print("Processing complete.")
