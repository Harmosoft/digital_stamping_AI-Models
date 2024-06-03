import os

# Function to preprocess data
def preprocess_data(fraudulent_dir, genuine_dir):
    fraudulent_files = [os.path.join(fraudulent_dir, f) for f in os.listdir(fraudulent_dir)]
    genuine_files = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir)]

    # Perform any necessary preprocessing steps here
    # For example, you may want to clean the text, remove stopwords, etc.

    return fraudulent_files, genuine_files

# Preprocess data
fraudulent_files, genuine_files = preprocess_data('data/fraudulent_documents', 'data/genuine_documents')
