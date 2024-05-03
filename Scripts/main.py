import numpy as np
from sklearn.model_selection import train_test_split


import os
from random import random

import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from LSTM_model import LSTM_model
from Bi_LSTM_model import Bi_LSTM_model

# RNN Model
from Bi_RNN_model import bi_rnn_model
from RNN_model import rnn_model

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

    # Save as a CSV
    # combined_df.to_csv('combined_data.csv', index=False)

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

        print(f'Epoch {epoch + 1}/{epochs} - Training MAE: {train_mae:.4f} - Training MSE: {train_mse:.4f} - Validation MAE: {val_mae:.4f} - Validation MSE: {val_mse:.4f}')

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

    print(f"Mean Absolute Error (MAE): {mae :.2f} \tMean Squared Error (MSE): {mse:.2f}")

    return mae, mse

def data_split(X_test, y_test):
    # [Short(below 800)]
    # JFK -> ATL(759)
    # JFK -> ORD(738)
    # ATL -> ORD(606)
    # ATL -> DFW(739)
    # DEN -> DFW(662)
    #
    # [Medium(800~1250)]
    # LAX -> DEN(860)
    # DEN -> ORD(806)
    # ORD -> DFW(801)
    # LAX -> DFW(1232)
    # ATL -> DEN(1196)
    #
    # [Long(over 1250)]
    # JFK -> LAX(2469)
    # JFK -> DFW(1388)
    # LAX -> DFW(1232)
    # LAX -> ORD(1741)
    # LAX -> ATL(1946)

    # 'JFK': 0, 'LAX': 1, 'DEN': 2, 'ATL': 3, 'DFW': 4, 'ORD': 5


    # Initialize empty DataFrames for each category
    short_distance_X = pd.DataFrame(columns=X_test.columns)
    short_distance_y = pd.Series()
    medium_distance_X = pd.DataFrame(columns=X_test.columns)
    medium_distance_y = pd.Series()
    long_distance_X = pd.DataFrame(columns=X_test.columns)
    long_distance_y = pd.Series()

    # Boolean indexing for short distance
    short_distance_conditions = (
        (X_test['Origin'] == 0) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 0) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 4) |

        (X_test['Origin'] == 3) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 3) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 2)
    )
    short_distance_X = X_test[short_distance_conditions]
    short_distance_y = y_test[short_distance_conditions]

    # Boolean indexing for medium distance
    medium_distance_conditions = (
        (X_test['Origin'] == 1) & (X_test['Destination'] == 2) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 2) |

        (X_test['Origin'] == 2) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 2) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 2) & (X_test['Destination'] == 3)
    )
    medium_distance_X = X_test[medium_distance_conditions]
    medium_distance_y = y_test[medium_distance_conditions]

    # Boolean indexing for long distance
    long_distance_conditions = (
        (X_test['Origin'] == 0) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 0) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 4) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 5) |
        (X_test['Origin'] == 1) & (X_test['Destination'] == 3) |

        (X_test['Origin'] == 1) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 0) |
        (X_test['Origin'] == 4) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 5) & (X_test['Destination'] == 1) |
        (X_test['Origin'] == 3) & (X_test['Destination'] == 1)
    )
    long_distance_X = X_test[long_distance_conditions]
    long_distance_y = y_test[long_distance_conditions]


    pd.set_option('display.max_columns', None)

    # Display the resulting DataFrames
    # print("Short Distance:")
    # print(short_distance_X.head())
    # print(short_distance_y.head())
    #
    # print("\nMedium Distance:")
    # print(medium_distance_X.head())
    # print(medium_distance_y.head())
    #
    # print("\nLong Distance:")
    # print(long_distance_X.head())
    # print(long_distance_y.head())

    X_test_dist = [short_distance_X, medium_distance_X, long_distance_X]
    y_test_dist = [short_distance_y, medium_distance_y, long_distance_y]


    return X_test_dist, y_test_dist


def main():
    # Load the dataset
    datasets, labels = load_dataset(data_dir)

    for i in range(len(datasets)):
        unique_values = datasets[i]['Travel Month'].unique()
        # print("unique_values", unique_values)

    # print(len(datasets))
    print(datasets[0].head())

    mae_list = []
    mse_list = []
    distance_type = ["Short Distance", "Medium Distance", "Long Distance"]

    for i in range(len(datasets)):
        print("\nWindow : ", i+1)

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

        X_test_dist, y_test_dist = data_split(X_test, y_test)

        # Model Preparation

        # Reshape X_train and X_val to add the timestep dimension
        X_train_reshaped = np.expand_dims(X_train, axis=1)
        X_val_reshaped = np.expand_dims(X_val, axis=1)
        X_test_reshaped = np.expand_dims(X_test, axis=1)

        # Print the shapes to verify
        print("X_train shape after reshaping:", X_train_reshaped.shape)
        print("X_val shape after reshaping:", X_val_reshaped.shape)
        print("X_test shape after reshaping:", X_test_reshaped.shape)

        input_shape = (len(X_train), X_train.shape[1],)  # Shape of input data for LSTM model

        # Train the model
        model = Bi_LSTM_model(input_shape)
        train_model(model, X_train_reshaped, y_train, X_val_reshaped, y_val, epochs=100)


        for i in range(len(X_test_dist)):
            X_test = X_test_dist[i]
            y_test = y_test_dist[i]
            X_test_reshaped = np.expand_dims(X_test, axis=1)

            # Evaluate the model
            print(distance_type[i])
            mae, mse = evaluate_model(model, X_test_reshaped, y_test)

            mae_list.append(mae)
            mse_list.append(mse)

    print("\n\nEvaluation results")
    for i in range(len(datasets)):
        print()
        for j in range(3): #for short, medium, long
            print(f"Window {i+1} {distance_type[j]}: \tMAE: {mae_list[i+j]:.2f} \tMSE: {mse_list[i+j]:.2f}")

if __name__ == '__main__':
    main()

