from keras.models import Sequential
from keras.layers import Bidirectional, SimpleRNN, Dense, Dropout


def create_bidirectional_rnn_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(128, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(SimpleRNN(64)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model
