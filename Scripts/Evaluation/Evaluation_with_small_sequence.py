import os
import numpy as np
import pandas as pd
from keras.src.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Scripts.Small_Sequence_Models.RNN_model import RNN_model
from Scripts.Small_Sequence_Models.Bidirectional_RNN_model import Bidirectional_RNN_model
from Scripts.Small_Sequence_Models.GRU_model import GRU_model
from Scripts.Small_Sequence_Models.Bidirectional_GRU_model import Bidirectional_GRU_model
from Scripts.Small_Sequence_Models.LSTM_model import LSTM_model
from Scripts.Small_Sequence_Models.Bidirectional_LSTM_model import Bidirectional_LSTM_model

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
        # Extract input features (excluding the target column)
        seq_x = np.delete(data[i:i+n_steps, :], target_column_index, axis=1)
        # Extract the target column
        seq_y = data[i+n_steps, target_column_index]
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

def plot_graphs(history_RNN, history_Bi_RNN, history_GRU, history_Bi_GRU, history_LSTM, history_Bi_LSTM):
    plt.figure(figsize=(20, 15))

    # Plot training and validation loss
    plt.subplot(3, 2, 1)
    plt.plot(history_RNN.history['loss'], label='RNN Train Loss')
    plt.plot(history_Bi_RNN.history['loss'], label='Bidirectional RNN Train Loss')
    plt.plot(history_GRU.history['loss'], label='GRU Train Loss')
    plt.plot(history_Bi_GRU.history['loss'], label='Bidirectional GRU Train Loss')
    plt.plot(history_LSTM.history['loss'], label='LSTM Train Loss')
    plt.plot(history_Bi_LSTM.history['loss'], label='Bidirectional LSTM Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(history_RNN.history['val_loss'], label='RNN Validation Loss')
    plt.plot(history_Bi_RNN.history['val_loss'], label='Bidirectional RNN Validation Loss')
    plt.plot(history_GRU.history['val_loss'], label='GRU Validation Loss')
    plt.plot(history_Bi_GRU.history['val_loss'], label='Bidirectional GRU Validation Loss')
    plt.plot(history_LSTM.history['val_loss'], label='LSTM Validation Loss')
    plt.plot(history_Bi_LSTM.history['val_loss'], label='Bidirectional LSTM Validation Loss')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation MAE
    plt.subplot(3, 2, 3)
    plt.plot(history_RNN.history['mae'], label='RNN Train MAE')
    plt.plot(history_Bi_RNN.history['mae'], label='Bidirectional RNN Train MAE')
    plt.plot(history_GRU.history['mae'], label='GRU Train MAE')
    plt.plot(history_Bi_GRU.history['mae'], label='Bidirectional GRU Train MAE')
    plt.plot(history_LSTM.history['mae'], label='LSTM Train MAE')
    plt.plot(history_Bi_LSTM.history['mae'], label='Bidirectional LSTM Train MAE')
    plt.title('Training MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(history_RNN.history['val_mae'], label='RNN Validation MAE')
    plt.plot(history_Bi_RNN.history['val_mae'], label='Bidirectional RNN Validation MAE')
    plt.plot(history_GRU.history['val_mae'], label='GRU Validation MAE')
    plt.plot(history_Bi_GRU.history['val_mae'], label='Bidirectional GRU Validation MAE')
    plt.plot(history_LSTM.history['val_mae'], label='LSTM Validation MAE')
    plt.plot(history_Bi_LSTM.history['val_mae'], label='Bidirectional LSTM Validation MAE')
    plt.title('Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    # Plot loss
    plt.subplot(3, 2, 5)
    plt.plot(history_RNN.history['loss'], label='RNN Train Loss')
    plt.plot(history_RNN.history['val_loss'], label='RNN Validation Loss')
    plt.plot(history_Bi_RNN.history['loss'], label='Bidirectional RNN Train Loss')
    plt.plot(history_Bi_RNN.history['val_loss'], label='Bidirectional RNN Validation Loss')
    plt.plot(history_GRU.history['loss'], label='GRU Train Loss')
    plt.plot(history_GRU.history['val_loss'], label='GRU Validation Loss')
    plt.plot(history_Bi_GRU.history['loss'], label='Bidirectional GRU Train Loss')
    plt.plot(history_Bi_GRU.history['val_loss'], label='Bidirectional GRU Validation Loss')
    plt.plot(history_LSTM.history['loss'], label='LSTM Train Loss')
    plt.plot(history_LSTM.history['val_loss'], label='LSTM Validation Loss')
    plt.plot(history_Bi_LSTM.history['loss'], label='Bidirectional LSTM Train Loss')
    plt.plot(history_Bi_LSTM.history['val_loss'], label='Bidirectional LSTM Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Plot MAE
    plt.subplot(3, 2, 6)
    plt.plot(history_RNN.history['mae'], label='RNN Train MAE')
    plt.plot(history_RNN.history['val_mae'], label='RNN Validation MAE')
    plt.plot(history_Bi_RNN.history['mae'], label='Bidirectional RNN Train MAE')
    plt.plot(history_Bi_RNN.history['val_mae'], label='Bidirectional RNN Validation MAE')
    plt.plot(history_GRU.history['mae'], label='GRU Train MAE')
    plt.plot(history_GRU.history['val_mae'], label='GRU Validation MAE')
    plt.plot(history_Bi_GRU.history['mae'], label='Bidirectional GRU Train MAE')
    plt.plot(history_Bi_GRU.history['val_mae'], label='Bidirectional GRU Validation MAE')
    plt.plot(history_LSTM.history['mae'], label='LSTM Train MAE')
    plt.plot(history_LSTM.history['val_mae'], label='LSTM Validation MAE')
    plt.plot(history_Bi_LSTM.history['mae'], label='Bidirectional LSTM Train MAE')
    plt.plot(history_Bi_LSTM.history['val_mae'], label='Bidirectional LSTM Validation MAE')
    plt.title('Mean Absolute Error (MAE)')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()



def main():
    current_directory = os.getcwd()
    base_directory = os.path.dirname(current_directory)
    preprocessed_folder = os.path.join(base_directory, '../preprocessed_data')
    output_folder = os.path.join(base_directory, '../combined_data')
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, 'combined_flight_data.csv')

    if os.path.exists(output_file_path):
        os.remove(output_file_path)


    # load the dataset and save the combined file
    df = load_data(preprocessed_folder)
    df.to_csv(output_file_path, index=False)
    print(f"Data saved to {output_file_path}")

    # print("df", df.head())


    # prepare datset for the deep learning models
    X, y = prepare_data(df, n_steps=20)

    # print("X", X[0])

    # spit the dataset
    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.5, random_state=42)



    input_shape = (X_train.shape[1], X_train.shape[2])

    # Define the model
    Uni_RNN_model = RNN_model(input_shape)
    Uni_RNN_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    Bi_RNN_model = Bidirectional_RNN_model(input_shape)
    Bi_RNN_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    Uni_GRU_model = GRU_model(input_shape)
    Uni_GRU_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    Bi_GRU_model = Bidirectional_GRU_model(input_shape)
    Bi_GRU_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    Uni_LSTM_model = LSTM_model(input_shape)
    Uni_LSTM_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    Bi_LSTM_model = Bidirectional_LSTM_model(input_shape)
    Bi_LSTM_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])


    # train models
    history_RNN = train_model(Uni_RNN_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    history_Bi_RNN = train_model(Bi_RNN_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    history_GRU = train_model(Uni_GRU_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    history_Bi_GRU = train_model(Bi_GRU_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    history_LSTM = train_model(Uni_LSTM_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)
    history_bi_LSTM = train_model(Bi_LSTM_model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32)


    # evaluation
    loss_RNN, mae_RNN, mse_RNN = evaluate_model(Uni_RNN_model, X_test, y_test)
    loss_Bi_RNN, mae_Bi_RNN, mse_Bi_RNN = evaluate_model(Bi_RNN_model, X_test, y_test)

    loss_GRU, mae_GRU, mse_GRU = evaluate_model(Uni_GRU_model, X_test, y_test)
    loss_Bi_GRU, mae_Bi_GRU, mse_Bi_GRU = evaluate_model(Bi_GRU_model, X_test, y_test)

    loss_LSTM, mae_LSTM, mse_LSTM = evaluate_model(Uni_LSTM_model, X_test, y_test)
    loss_Bi_LSTM, mae_Bi_LSTM, mse_Bi_LSTM = evaluate_model(Bi_LSTM_model, X_test, y_test)

    print("RNN Model - Loss:", loss_RNN, "MAE:", mae_RNN, "MSE:", mse_RNN)
    print("Bidirectional RNN Model - Loss:", loss_Bi_RNN, "MAE:", mae_Bi_RNN, "MSE:", mse_Bi_RNN)

    print("GRU Model - Loss:", loss_GRU, "MAE:", mae_GRU, "MSE:", mse_GRU)
    print("Bidirectional GRU Model - Loss:", loss_Bi_GRU, "MAE:", mae_Bi_GRU, "MSE:", mse_Bi_GRU)

    print("LSTM Model - Loss:", loss_LSTM, "MAE:", mae_LSTM, "MSE:", mse_LSTM)
    print("Bidirectional LSTM Model - Loss:", loss_Bi_LSTM, "MAE:", mae_Bi_LSTM, "MSE:", mse_Bi_LSTM)


    # plot graphs
    plot_graphs(history_RNN, history_Bi_RNN, history_GRU, history_Bi_GRU, history_LSTM, history_bi_LSTM)

if __name__ == "__main__":
    main()
