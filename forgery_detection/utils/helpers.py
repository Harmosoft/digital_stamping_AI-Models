import numpy as np

# Function to load data from CSV file
def load_data_from_csv(file_path):
    data = np.loadtxt(file_path, delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]
    return X, y

# Function to save data to CSV file
def save_data_to_csv(X, y, file_path):
    data = np.column_stack((X, y))
    np.savetxt(file_path, data, delimiter=',')
