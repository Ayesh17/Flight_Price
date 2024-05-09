from keras.models import Sequential
from keras.layers import Bidirectional, SimpleRNN, Dense, Dropout, Input


def Bidirectional_RNN_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(SimpleRNN(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
