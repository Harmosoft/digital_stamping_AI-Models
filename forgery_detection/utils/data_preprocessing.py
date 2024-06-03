import cv2
import os

# Define paths to genuine and forged signature samples
genuine_dir = 'data/genuine_signatures'
forged_dir = 'data/forged_signatures'

# Function to preprocess image data
def preprocess_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append(img)
        labels.append(1 if directory == genuine_dir else 0)
    return np.array(data), np.array(labels)

# Preprocess genuine signature samples
genuine_data, genuine_labels = preprocess_data(genuine_dir)

# Preprocess forged signature samples
forged_data, forged_labels = preprocess_data(forged_dir)
