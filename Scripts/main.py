import os
from random import random

import numpy as np
import pandas as pd
import pickle

from keras import Input
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile, f_classif, RFE
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout, SimpleRNN, GRU
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random
import tensorflow as tf
import keras
from keras.callbacks import TensorBoard, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC  # You may need to choose an appropriate estimator for your problem
from sklearn.tree import DecisionTreeClassifier
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
import LSTM_model


# Folder structure
data_dir = 'Preprocessed_data'


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

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

    # # Concatenate all DataFrames into a single DataFrame
    # combined_df = pd.concat(dfs, ignore_index=True)

    # Concatenate all DataFrames into a single DataFrame based on 'travel_date' column
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Travel Day of Year')

    print("combined_df", combined_df.head())

    column_names = combined_df.columns.tolist()

    # Save as a CSV
    combined_df.to_csv('combined_data.csv', index=False)

    # Get the dataset and labels
    dataset = combined_df.drop(columns=['Price ($)'])
    labels = combined_df['Price ($)']

    print("dataset", dataset.shape)
    print("labels", labels.shape)

    return dataset, labels


def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs):
    model.compile(loss='mean_squared_error', optimizer='adam')  # Using mean squared error as loss for regression
    lowest_val_loss = float('inf')  # Initialize with a large value
    best_weights = None

    for epoch in range(epochs):
        history = model.fit(X_train, y_train, epochs=1, verbose=0)

        # Calculate training predictions
        y_train_pred = model.predict(X_train)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)

        # Calculate validation predictions
        y_val_pred = model.predict(X_val)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        print(
            f'Epoch {epoch + 1}/{epochs} - Training MAE: {train_mae:.4f} - Training MSE: {train_mse:.4f} - Validation MAE: {val_mae:.4f} - Validation MSE: {val_mse:.4f}')

        # Calculate validation loss
        val_loss = model.evaluate(X_val, y_val, verbose=0)

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_weights = model.get_weights()

    model.set_weights(best_weights)

# Initialize lists to store split data from each window
combined_train_data = []
combined_validation_data = []
combined_test_data = []

# Existing loop to process windowed data
for window_data in windowed_datasets:
    # Define the split indices
    train_end = int(len(window_data) * 0.6)  # 60% of the data for training
    validation_end = train_end + int(len(window_data) * 0.2)  # Additional 20% for validation

    # Split the data sequentially
    train_data = window_data[:train_end]
    validation_data = window_data[train_end:validation_end]
    test_data = window_data[train_end:]  # Test data starts from the beginning of validation and goes to the end

    # Append to lists
    combined_train_data.append(train_data)
    combined_validation_data.append(validation_data)
    combined_test_data.append(test_data)

def evaluate_model(model, X_test, y_test):
    # Compute predictions
    y_pred = model.predict(X_test)

    # Compute MAE and MSE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)


def main():
    # Load the dataset
    dataset, labels = load_dataset(data_dir)


    X_train, X_val, y_train, y_val = train_test_split(dataset, labels, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.50, random_state=42)
    print("X_train", len(X_train))
    print("X_val", len(X_val))
    print("X_test", len(X_test))


    # # Convert labels to categorical format
    # num_classes = 3  # Number of classes
    # y_train = to_categorical(y_train, num_classes)
    # y_val = to_categorical(y_val, num_classes)
    # y_test = to_categorical(y_test, num_classes)
    #
    # Create the model
    print("X_train shape:", X_train.shape[1])

    # Reshape X_train and X_val to add the timestep dimension
    X_train_reshaped = np.expand_dims(X_train, axis=1)
    X_val_reshaped = np.expand_dims(X_val, axis=1)
    X_test_reshaped = np.expand_dims(X_test, axis=1)

    # Print the shapes to verify
    print("X_train shape after reshaping:", X_train_reshaped.shape)
    print("X_val shape after reshaping:", X_val_reshaped.shape)
    print("X_test shape after reshaping:", X_test_reshaped.shape)

    input_shape = (len(X_train), X_train.shape[1],)   # Shape of input data for LSTM model
    print(input_shape)

    #Updae this based on the model we use
    model = LSTM_model.create_model(input_shape)

    # Train the model
    train_model(model, X_train_reshaped, y_train, X_val_reshaped, y_val, epochs=1000)


    # Evaluate the model
    evaluate_model(model, X_test_reshaped, y_test)


if __name__ == '__main__':
    main()

