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



# ...

# Folder structure
data_dir = 'Extracted_data'


# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def load_dataset(data_dir):
    # Get the current working directory
    current_directory = os.getcwd()
    base_directory = os.path.dirname(current_directory)
    input_directory = os.path.join(base_directory, data_dir)
    # print("input_directory", input_directory)

    # List to store DataFrames
    dfs = []

    # Iterate over each file in the current directory
    for file_name in os.listdir(input_directory):
        if file_name.endswith(".csv"):  # Check if the file is a CSV file
            file_path = os.path.join(input_directory, file_name)  # Construct the file path
            # print("file_path", file_path)
            df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame
            dfs.append(df)  # Append the DataFrame to the list

    # Concatenate all DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)
    # print("combined_df", combined_df.head())

    combined_df['Access Date'] = pd.to_datetime(combined_df['Access Date'])
    combined_df['Departure datetime'] = pd.to_datetime(combined_df['Departure datetime'])
    combined_df['Arrival datetime'] = pd.to_datetime(combined_df['Arrival datetime'])

    # Get the overall time of the flight
    combined_df['Time to Travel'] = (combined_df['Arrival datetime'] - combined_df[
        'Departure datetime']).dt.total_seconds() / 3600
    combined_df['Time to Travel'] = combined_df['Time to Travel'].round(2) # Round the time to travel to two decimal points

    # Get the time difference from access date to travel date
    # Calculate the difference in days between 'Access Date' and 'Departure Date'
    # Extract date part from 'Departure datetime' column
    # combined_df['Departure Date'] = combined_df['Departure datetime'].dt.date
    # combined_df['Access-Departure Days'] = (combined_df['Departure Date'] - combined_df['Access Date']).dt.days

    # Extract date of the year (day of the year) and hour components
    combined_df['Travel Day of Year'] = combined_df['Departure datetime'].dt.dayofyear
    combined_df['Travel Hour'] = combined_df['Departure datetime'].dt.hour
    combined_df['Access Day of Year'] = combined_df['Departure datetime'].dt.dayofyear

    # Drop instances where airline is wrong
    # Define a regular expression pattern to match non-letter characters or spaces
    pattern = r'[^a-zA-Z ]'
    # Filter the DataFrame to keep only rows where 'Airlines' column doesn't contain the pattern
    combined_df = combined_df[~combined_df['Airline(s)'].str.contains(pattern)]

    # Drop the original 'Arrival datetime' and 'Departure datetime' columns
    combined_df.drop(columns=['Arrival datetime'], inplace=True)
    combined_df.drop(columns=['Departure datetime'], inplace=True)
    # combined_df.drop(columns=['Departure date'], inplace=True)
    combined_df.drop(columns=['Access Date'], inplace=True)
    combined_df.drop(columns=['Travel Time'], inplace=True)
    combined_df.drop(columns=['Layover'], inplace=True)
    combined_df.drop(columns=['Airline(s)'], inplace=True)

    #convert non numerical values to numerical
    # Initialize LabelEncoder
    label_encoder = LabelEncoder()

    combined_df['Origin'] = label_encoder.fit_transform(combined_df['Origin'])
    combined_df['Destination'] = label_encoder.fit_transform(combined_df['Destination'])

    print("combined_df", combined_df.head())

    column_names = combined_df.columns.tolist()
    print("Column names:", column_names)

    #drop rows with incomplete data
    combined_df.dropna(inplace=True)

    #save as a csv
    combined_df.to_csv('combined_data.csv', index=False)

    #get the dataset and labels
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

    # Print the shapes to verify
    print("X_train shape after reshaping:", X_train_reshaped.shape)
    print("X_val shape after reshaping:", X_val_reshaped.shape)

    input_shape = (len(X_train), X_train.shape[1],)   # Shape of input data for LSTM model
    print(input_shape)
    model = create_model(input_shape)

    # Train the model
    train_model(model, X_train_reshaped, y_train, X_val_reshaped, y_val, epochs=100)


    # Evaluate the model
    evaluate_model(model, X_test, y_test)


if __name__ == '__main__':
    main()

