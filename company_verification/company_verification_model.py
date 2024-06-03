import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Load company profiles from JSON file
with open('data/company_profiles.json', 'r') as file:
    company_profiles = json.load(file)

# Load trusted sources from CSV file
trusted_sources = pd.read_csv('data/trusted_sources.csv')

# Define hyperparameters
NUM_CLASSES = 2
NUM_FEATURES = len(company_profiles[0]['features'])

# Preprocess company profiles
X = []
y = []
for profile in company_profiles:
    X.append(profile['features'])
    y.append(profile['label'])

X = np.array(X)
y = np.array(y)

# Split data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(NUM_FEATURES,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the trained model
model.save('company_verification_model.h5')
