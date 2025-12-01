# model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

def build_lstm_model(input_shape, units=64, dropout=0.2, output_dim=1, task='regression'):
    model = Sequential()
    model.add(LSTM(units, input_shape=input_shape))
    model.add(Dropout(dropout))
    if task == 'regression':
        model.add(Dense(output_dim))  # linear
        loss = 'mse'
        metrics = ['mae']
    else:
        # classification (binary or multi-class)
        if output_dim == 1:
            model.add(Dense(1, activation='sigmoid'))
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(Dense(output_dim, activation='softmax'))
            loss = 'sparse_categorical_crossentropy'
            metrics = ['accuracy']
    model.compile(optimizer='adam', loss=loss, metrics=metrics)
    return model

def save_model(model, path):
    model.save(path)

def load_trained_model(path):
    return load_model(path)
