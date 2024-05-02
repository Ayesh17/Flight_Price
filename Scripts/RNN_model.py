from keras.models import Sequential
from keras.layers import SimpleRNN, Dense, Dropout, Reshape, Input


def rnn_model(input_shape):
    model = Sequential()
    model.add(SimpleRNN(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(SimpleRNN(64))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

