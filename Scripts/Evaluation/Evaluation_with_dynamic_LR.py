import numpy as np
import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import random
from Scripts.Long_Sequence_Models.RNN_model import RNN_model
from Scripts.Long_Sequence_Models.Bidirectional_RNN_model import Bidirectional_RNN_model
from Scripts.Long_Sequence_Models.GRU_model import GRU_model
from Scripts.Long_Sequence_Models.Bidirectional_GRU_model import Bidirectional_GRU_model
from Scripts.Long_Sequence_Models.LSTM_model import LSTM_model
from Scripts.Long_Sequence_Models.Bidirectional_LSTM_model import Bidirectional_LSTM_model

# Folder structure
data_dir = '../Preprocessed_data'

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
    window_size_months = 8

    # Get the minimum and maximum flight months
    min_flight_month = combined_df['Travel Month'].min()
    max_flight_month = combined_df['Travel Month'].max()

    # List to store windowed datasets and labels
    windowed_datasets = []
    windowed_labels = []

    # Iterate over the flight months
    start_month = min_flight_month
    while start_month + window_size_months - 1 <= max_flight_month:
        end_month = start_month + window_size_months

        # Filter data for the current window
        window_data = combined_df[
            (combined_df['Travel Month'] >= start_month) & (combined_df['Travel Month'] < end_month)]

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
    # Dynamic learning rate scheduling
    reduce_lr = ReduceLROnPlateau(factor=0.1, patience=10, min_lr=1e-6, verbose=1)

    # Early stopping
    early_stop = EarlyStopping(patience=25, verbose=1, restore_best_weights=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Set initial learning rate
    model.compile(loss='mean_squared_error', optimizer=optimizer)  # Using mean squared error as loss for regression
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stop])


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

    overall_X = X_test
    overall_y = y_test

    X_test_dist = [short_distance_X, medium_distance_X, long_distance_X, overall_X]
    y_test_dist = [short_distance_y, medium_distance_y, long_distance_y, overall_y]

    return X_test_dist, y_test_dist

def main():
    # Load the dataset
    datasets, labels = load_dataset(data_dir)

    mae_list = []
    mse_list = []
    distance_type = ["Short Distance", "Medium Distance", "Long Distance", "Overall"]

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

        input_shape = (len(X_train), X_train.shape[1],)  # Shape of input data for LSTM model

        # Train the model
        model = LSTM_model(input_shape)
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

    count = 0
    print("\n\nEvaluation results")
    for i in range(len(datasets)):
        print()
        for j in range(len(distance_type)):  # for short, medium, long
            print(f"Window {i + 1} {distance_type[j]}: \tMAE: {mae_list[count]:.2f} \tMSE: {mse_list[count]:.2f}")
            count += 1


if __name__ == '__main__':
    main()
