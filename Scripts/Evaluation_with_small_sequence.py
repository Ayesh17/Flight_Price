import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Scripts.Small_Sequence_Models.LSTM_Model_2 import LSTM_model
from Bidirectional_GRU_model import Bidirectional_GRU_model

def load_data(directory_path):
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    df_list = []
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        df_temp = pd.read_csv(file_path)
        df_list.append(df_temp)
    df = pd.concat(df_list, ignore_index=True)
    return df

def create_sequences(data, target_column_index, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        seq_x, seq_y = data[i:i+n_steps, :], data[i+n_steps, target_column_index]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def prepare_data(df, n_steps):
    target_column_index = df.columns.get_loc('Price ($)')
    data_array = df.values
    X, y = create_sequences(data_array, target_column_index, n_steps)
    return X, y

def split_data(X, y):
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history

def evaluate_model(model, X_test, y_test):
    loss, mae = model.evaluate(X_test, y_test)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f"loss: {loss :.2f} \t  Mean Absolute Error (MAE): {mae :.2f} \tMean Squared Error (MSE): {mse:.2f}")
    return loss, mae, mse

def plot_loss(history_original, history_bidirectional):
    plt.figure(figsize=(15, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history_original.history['loss'], label='Original Model Train Loss')
    plt.plot(history_original.history['val_loss'], label='Original Model Validation Loss')
    plt.plot(history_bidirectional.history['loss'], label='Bidirectional Model Train Loss')
    plt.plot(history_bidirectional.history['val_loss'], label='Bidirectional Model Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history_original.history['mae'], label='Original Model Train MAE')
    plt.plot(history_original.history['val_mae'], label='Original Model Validation MAE')
    plt.plot(history_bidirectional.history['mae'], label='Bidirectional Model Train MAE')
    plt.plot(history_bidirectional.history['val_mae'], label='Bidirectional Model Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    current_directory = os.getcwd()
    base_directory = os.path.dirname(current_directory)
    input_folder = os.path.join(base_directory, 'extracted_data')
    preprocessed_folder = os.path.join(base_directory, 'preprocessed_data')
    output_folder = os.path.join(base_directory, 'combined_data')
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, 'combined_flight_data.csv')

    if os.path.exists(output_file_path):
        os.remove(output_file_path)


    # load the dataset and save the combined file
    df = load_data(preprocessed_folder)
    df.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")


    # prepare datset for the deep learning models
    X, y = prepare_data(df, n_steps=100)

    # spit the dataset
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define the model
    model_original = LSTM_model(input_shape)
    model_bidirectional = Bidirectional_GRU_model(input_shape)

    # train models
    history_original = train_model(model_original, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)
    history_bidirectional = train_model(model_bidirectional, X_train, y_train, X_val, y_val, epochs=10, batch_size=32)


    # evaluate
    loss_original, mae_original, mse_original  = evaluate_model(model_original, X_test, y_test)
    loss_bidirectional, mae_bidirectional, mse_bidirectional = evaluate_model(model_bidirectional, X_test, y_test)
    print("Original Model - Loss:", loss_original, "MAE:", mae_original, "MSE:", mse_original)
    print("Bidirectional Model - Loss:", loss_bidirectional, "MAE:", mae_bidirectional, "MSE:", mse_bidirectional)


    # plot graphs
    plot_loss(history_original, history_bidirectional)

if __name__ == "__main__":
    main()
