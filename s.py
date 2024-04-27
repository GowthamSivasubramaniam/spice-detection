import os
import shutil
import random

# Set the source directory
source_dir = 'Plants'

# Set the destination directories
train_dir = 'Dataset1/Train'
test_dir = 'Dataset1/Test'

# Get a list of class labels (subdirectories)
class_labels = os.listdir(source_dir)

test_percentage = 0.2

# Iterate through each class label
for label in class_labels:
    # Create destination directories if they don't exist
    os.makedirs(os.path.join(train_dir, label), exist_ok=True)
    os.makedirs(os.path.join(test_dir, label), exist_ok=True)

    # Get a list of image files in the current class label directory
    images = [file for file in os.listdir(os.path.join(source_dir, label)) if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    # Shuffle the list of images randomly
    random.shuffle(images)

    # Calculate the number of images to move to the test set
    num_test_images = int(len(images) * test_percentage)

    # Split the images into training and testing sets
    train_images = images[num_test_images:]
    test_images = images[:num_test_images]

    # Move images to the respective directories
    for image in train_images:
        src_path = os.path.join(source_dir, label, image)
        dst_path = os.path.join(train_dir, label, image)
        shutil.copy(src_path, dst_path)

    for image in test_images:
        src_path = os.path.join(source_dir, label, image)
        dst_path = os.path.join(test_dir, label, image)
        shutil.copy(src_path, dst_path)
