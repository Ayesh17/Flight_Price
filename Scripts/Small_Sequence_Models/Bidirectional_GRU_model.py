import numpy as np
import pandas as pd
from keras import Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import GRU, Input, Dense, Dropout, Bidirectional


def Bidirectional_GRU_model(input_shape):
    model = Sequential()
    model.add(Bidirectional(GRU(50, return_sequences=True), input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Bidirectional(GRU(50)))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model
