import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from RNN_model import rnn_model
from Bi_RNN_model import bi_rnn_model
import tensorflow as tf
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_error

matplotlib.use('TkAgg')
np.random.seed(42)
tf.random.set_seed(42)


def load_dataset(data_dir):
    dfs = []

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".csv"):
            file_path = os.path.join(data_dir, file_name)
            df = pd.read_csv(file_path)
            dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sort_values(by=['Travel Day of Year', 'Travel Hour'], inplace=True)
    window_size_months = 4
    min_flight_month = combined_df['Travel Month'].min()
    max_flight_month = combined_df['Travel Month'].max()
    windowed_datasets = []
    windowed_labels = []

    for start_month in range(min_flight_month, max_flight_month + 1 - window_size_months):
        end_month = start_month + window_size_months
        window_data = combined_df[
            (combined_df['Travel Month'] >= start_month) & (combined_df['Travel Month'] < end_month)]
        windowed_datasets.append(window_data.drop(columns=['Price ($)']).values)
        windowed_labels.append(window_data['Price ($)'].values)

    return windowed_datasets, windowed_labels


def train_model(model, X_train, y_train, X_val, y_val, epochs):
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, verbose=1, validation_data=(X_val, y_val))

    return history


def data_split(X_test, y_test):
    # Assume the structure of X_test is known and fixed
    # Index 0 for Origin, Index 1 for Destination

    # Define the conditions using indices
    short_distance_conditions = (
            (X_test[:, 0] == 0) & (X_test[:, 1] == 3) |  # JFK -> ATL
            (X_test[:, 0] == 0) & (X_test[:, 1] == 5) |  # JFK -> ORD
            (X_test[:, 0] == 3) & (X_test[:, 1] == 5) |  # ATL -> ORD
            (X_test[:, 0] == 3) & (X_test[:, 1] == 4) |  # ATL -> DFW
            (X_test[:, 0] == 2) & (X_test[:, 1] == 4)  # DEN -> DFW
    )

    medium_distance_conditions = (
            (X_test[:, 0] == 1) & (X_test[:, 1] == 2) |  # LAX -> DEN
            (X_test[:, 0] == 2) & (X_test[:, 1] == 5) |  # DEN -> ORD
            (X_test[:, 0] == 5) & (X_test[:, 1] == 4) |  # ORD -> DFW
            (X_test[:, 0] == 1) & (X_test[:, 1] == 4) |  # LAX -> DFW
            (X_test[:, 0] == 3) & (X_test[:, 1] == 2)  # ATL -> DEN
    )

    long_distance_conditions = (
            (X_test[:, 0] == 0) & (X_test[:, 1] == 1) |  # JFK -> LAX
            (X_test[:, 0] == 0) & (X_test[:, 1] == 4) |  # JFK -> DFW
            (X_test[:, 0] == 1) & (X_test[:, 1] == 4) |  # LAX -> DFW
            (X_test[:, 0] == 1) & (X_test[:, 1] == 5) |  # LAX -> ORD
            (X_test[:, 0] == 1) & (X_test[:, 1] == 3)  # LAX -> ATL
    )

    # Apply conditions
    short_distance_X = X_test[short_distance_conditions]
    short_distance_y = y_test[short_distance_conditions]
    medium_distance_X = X_test[medium_distance_conditions]
    medium_distance_y = y_test[medium_distance_conditions]
    long_distance_X = X_test[long_distance_conditions]
    long_distance_y = y_test[long_distance_conditions]

    # Return the split data
    X_test_dist = [short_distance_X, medium_distance_X, long_distance_X]
    y_test_dist = [short_distance_y, medium_distance_y, long_distance_y]

    return X_test_dist, y_test_dist


def plot_learning_curves(history, filename, show_fig=False):
    plt.figure(figsize=(12, 6))
    plt.suptitle(filename, fontsize=16)
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()

    plt.savefig(filename)


def evaluate_model(model, X_test, y_test):
    # Compute predictions
    y_pred = model.predict(X_test)

    # Compute MAE and MSE
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae :.2f} \tMean Squared Error (MSE): {mse:.2f}")

    return mae, mse


def main(use_bidirectional=False):
    # Load the dataset
    data_dir = '../preprocessed_data'
    datasets, labels = load_dataset(data_dir)

    mae_list = []
    mse_list = []
    distance_type = ["Short Distance", "Medium Distance", "Long Distance"]

    for i in range(len(datasets)):
        print("\nWindow : ", i + 1)

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

        model_type = ''
        # Choose the model based on the use_bidirectional flag
        if use_bidirectional:
            model = bi_rnn_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
            model_type = 'Bi-RNN'
        else:
            model = rnn_model((X_train_reshaped.shape[1], X_train_reshaped.shape[2]))
            model_type = 'RNN'

        # Train the model
        history = train_model(model, X_train_reshaped, y_train, X_val_reshaped, y_val, epochs=100)
        plot_learning_curves(history, f'learning_curve_{model_type}_window_{i + 1}.png', model_type)

        for j in range(len(X_test_dist)):
            X_test = X_test_dist[j]
            y_test = y_test_dist[j]
            X_test_reshaped = np.expand_dims(X_test, axis=1)

            # Evaluate the model
            print(distance_type[j])
            mae, mse = evaluate_model(model, X_test_reshaped, y_test)

            mae_list.append(mae)
            mse_list.append(mse)

    print("\n\nEvaluation results")
    for i in range(len(datasets)):
        print()
        for j in range(3):  # for short, medium, long
            print(
                f"Window {i + 1} {distance_type[j]}: \tMAE: {mae_list[3 * i + j]:.2f} \tMSE: {mse_list[3 * i + j]:.2f}")


if __name__ == '__main__':
    # Change to 'False' to use RNN or 'True' to use Bi-RNN
    main(use_bidirectional=False)
