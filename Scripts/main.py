import numpy as np
from sklearn.model_selection import train_test_split


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
from LSTM_model import LSTM_model

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

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True).sort_values(by='Travel Day of Year')

    # Sort the DataFrame by 'Travel Day of Year' and 'Travel Hour'
    combined_df = combined_df.sort_values(by=['Travel Day of Year', 'Travel Hour'])

    # Define the window size in terms of months
    window_size_months = 4

    # Get the minimum and maximum flight months
    min_flight_month = combined_df['Travel Month'].min()
    max_flight_month = combined_df['Travel Month'].max()

    # List to store windowed datasets and labels
    windowed_datasets = []
    windowed_labels = []

    # Iterate over the flight months
    start_month = min_flight_month
    while start_month + window_size_months -1 <= max_flight_month:
        end_month = start_month + window_size_months

        # Filter data for the current window
        window_data = combined_df[(combined_df['Travel Month'] >= start_month) & (combined_df['Travel Month'] < end_month)]

        # Extract features and labels
        window_dataset = window_data.drop(columns=['Price ($)'])
        window_labels = window_data['Price ($)']

        # Append to the list
        windowed_datasets.append(window_dataset)
        windowed_labels.append(window_labels)

        # Move to the next window
        start_month += 1

    return windowed_datasets, windowed_labels


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


def evaluate_model(model, X_test, y_test):
    # Compute predictions
    y_pred = model.predict(X_test)

    # Compute MAE and MSE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print('Mean Absolute Error (MAE):', mae)
    print('Mean Squared Error (MSE):', mse)


# def generate_rolling_windows(dataset, labels, window_size=120, train_ratio=0.7, val_ratio=0.15):
#     num_windows = len(dataset) - window_size + 1
#     windows = []
#     for i in range(num_windows):
#         window_data = dataset[i:i + window_size]
#         window_labels = labels[i:i + window_size]
#         X_train, X_temp, y_train, y_temp = train_test_split(window_data, window_labels, train_size=train_ratio)
#         X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
#         windows.append((X_train, X_val, X_test, y_train, y_val, y_test))
#     return windows



def main():
    # Load the dataset
    print("window")
    datasets, labels = load_dataset(data_dir)

    for i in range(len(datasets)):
        unique_values = datasets[i]['Travel Month'].unique()
        print("unique_values", unique_values)

    print(len(datasets))
    print(datasets[0].head())
    print(datasets[0].tail())

    for i in range(len(datasets)):
        # Split the windowed data into train, validation, and test sets (80-10-10 split)
        data = datasets[i]
        label = labels[i]
        train_size = int(0.8 * len(data))
        val_size = int(0.1 * len(data))

        X_train = data[:train_size]
        y_train = label[:train_size]

        X_val = data[train_size:train_size + val_size]
        y_val = label[train_size:train_size + val_size]

        X_test = data[train_size + val_size:]
        y_test = label[train_size + val_size:]

        # Model Preparation
        input_shape = (X_train.shape[1],)  # Shape of input data for LSTM model



        # Train the model
        model = LSTM_model(input_shape)
        train_model(model, X_train, y_train, X_val, y_val, epochs=2)

        # Evaluate the model
        evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()

