import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Define paths to genuine and forged signature samples
genuine_dir = 'data/genuine_signatures'
forged_dir = 'data/forged_signatures'

# Function to load and preprocess image data
def load_data(directory):
    data = []
    labels = []
    for filename in os.listdir(directory):
        img_path = os.path.join(directory, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        data.append(img)
        labels.append(1 if directory == genuine_dir else 0)
    return np.array(data), np.array(labels)

# Load and preprocess image data
genuine_data, genuine_labels = load_data(genuine_dir)
forged_data, forged_labels = load_data(forged_dir)

# Concatenate data and labels
X = np.concatenate((genuine_data, forged_data), axis=0)
y = np.concatenate((genuine_labels, forged_labels), axis=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

# Load MobileNetV2 model without top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Add custom classification head
head_model = base_model.output
head_model = AveragePooling2D(pool_size=(7, 7))(head_model)
head_model = Flatten(name='flatten')(head_model)
head_model = Dense(128, activation='relu')(head_model)
head_model = Dropout(0.5)(head_model)
head_model = Dense(2, activation='softmax')(head_model)

# Combine base model with custom head
model = Model(inputs=base_model.input, outputs=head_model)

# Freeze layers in base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
opt = Adam(lr=1e-4)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_test, y_test))

# Evaluate the model
predictions = model.predict(X_test, batch_size=32)
print(classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1)))

# Save the trained model
model.save('forgery_detection_model.h5')
