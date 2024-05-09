from keras.models import Sequential
from keras.layers import Bidirectional, SimpleRNN, Dense, Dropout, Input


def Bidirectional_RNN_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(SimpleRNN(128, return_sequences=True)))
    model.add(Dropout(0.2))
    model.add(Bidirectional(SimpleRNN(64)))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model

