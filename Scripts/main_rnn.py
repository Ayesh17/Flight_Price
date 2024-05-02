import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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


def main(use_bidirectional=True):
    data_dir = '../preprocessed_data'
    datasets, labels = load_dataset(data_dir)
    mae_list = []
    mse_list = []
    results = []

    for idx, (data, label) in enumerate(zip(datasets, labels)):
        X_train, X_val, y_train, y_val = train_test_split(data, label, test_size=0.2, random_state=42)
        X_train = np.expand_dims(X_train, axis=1)
        X_val = np.expand_dims(X_val, axis=1)
        model_type = 'Bi-RNN' if use_bidirectional else 'RNN'
        model = bi_rnn_model((1, X_train.shape[2])) if use_bidirectional else rnn_model((1, X_train.shape[2]))
        history = train_model(model, X_train, y_train, X_val, y_val, epochs=100)
        plot_learning_curves(history, f'learning_curve_{model_type}_window_{idx + 1}.png', model_type)

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        mae_list.append(mae)
        mse_list.append(mse)
        results.append(f"Window {idx + 1}: MAE = {mae:.2f}, MSE = {mse:.2f}")

    print("\n".join(results))
    print("Average MAE:", np.mean(mae_list))
    print("Average MSE:", np.mean(mse_list))


if __name__ == '__main__':
    main()
