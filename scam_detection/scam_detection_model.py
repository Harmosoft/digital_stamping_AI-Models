import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Load data
fraudulent_dir = 'data/fraudulent_documents'
genuine_dir = 'data/genuine_documents'

fraudulent_files = [os.path.join(fraudulent_dir, f) for f in os.listdir(fraudulent_dir)]
genuine_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)]

documents = []
labels = []

# Read fraudulent documents
for file in fraudulent_files:
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())
        labels.append(1)  # Fraudulent

# Read genuine documents
for file in genuine_files:
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())
        labels.append(0)  # Genuine

# Convert documents to TF-IDF vectors
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(documents)
y = np.array(labels)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save trained model
joblib.dump(model, 'scam_detection_model.pkl')
