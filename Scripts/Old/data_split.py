import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras import Input
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.src.layers import Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define the custom data loader
class DataLoader(tf.keras.utils.Sequence):
    def __init__(self, dataset, labels, window_size, batch_size):
        self.dataset = dataset
        self.labels = labels
        self.window_size = window_size
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset) - self.window_size + 1)

    def __len__(self):
        return len(self.indexes) // self.batch_size

    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_data = []
        batch_labels = []
        for i in batch_indexes:
            window_data = self.dataset[i:i + self.window_size]
            window_labels = self.labels[i:i + self.window_size]
            batch_data.append(window_data)
            batch_labels.append(window_labels[-1])  # Using the last label in the window as the target
        return np.array(batch_data), np.array(batch_labels)



def LSTM_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model


# Load the dataset
def load_dataset(data_dir):
    # Get the current working directory
    current_directory = os.getcwd()
    base_directory = os.path.dirname(current_directory)
    input_directory = os.path.join(base_directory, data_dir)

    # List to store DataFrames
    dfs = []

    # Iterate over each file in the current directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):  # Check if the file is a CSV file
            file_path = os.path.join(input_directory, file_name)  # Construct the file path
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
            dfs.append(df)  # Append the DataFrame to the list

    # Concatenate all DataFrames into a single DataFrame based on 'travel_date' column
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Travel Day of Year')

    # Sort the DataFrame by 'Travel Day of Year' and 'Travel Hour'
    combined_df = combined_df.sort_values(by=['Travel Day of Year', 'Travel Hour'])

    combined_df = combined_df.head(640)

    # Get the dataset and labels
    dataset = combined_df.drop(columns=['Price ($)']).values
    labels = combined_df['Price ($)'].values

    return dataset, labels

# Train the model
def train_model(model, data_loader, epochs):
    model.compile(loss='mean_squared_error', optimizer='adam')  # Using mean squared error as loss for regression
    model.fit(data_loader, epochs=epochs, verbose=1)

# Evaluate the model
def evaluate_model(model, test_loader):
    # Compute predictions
    y_pred = model.predict(test_loader)
    print("y sizes", y_pred)

    # Flatten the predictions
    y_pred = y_pred.flatten()
    print("y sizes", y_pred)

    # Compute MAE and MSE
    mae = mean_absolute_error(test_loader.labels, y_pred)
    mse = mean_squared_error(test_loader.labels, y_pred)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)


def main():
    # Load the dataset
    dataset, labels = load_dataset('Preprocessed_data')

    # Define parameters
    window_size = 120
    batch_size = 64
    epochs = 2

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(dataset, labels, test_size=0.2, random_state=42)

    print("sizes", X_train.shape, X_test.shape)

    # Create data loader for training
    train_loader = DataLoader(X_train, y_train, window_size, batch_size)

    # Create data loader for testing
    test_loader = DataLoader(X_test, y_test, window_size, batch_size)

    # Model Preparation
    input_shape = (window_size, X_train.shape[1])  # Shape of input data for LSTM model

    # Build and train the model
    model = LSTM_model(input_shape)
    train_model(model, train_loader, epochs)

    # Evaluate the model
    evaluate_model(model, test_loader)


if __name__ == '__main__':
    main()
