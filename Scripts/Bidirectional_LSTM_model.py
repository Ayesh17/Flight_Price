def create_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))  # Using Bidirectional LSTM
    model.add(Dropout(0.1))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))
    return model
