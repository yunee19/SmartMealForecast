import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Load data
X = pd.read_csv("X_encoded.csv")
y = pd.read_csv("y_lunch.csv").values.ravel()

# Reshape for LSTM
X = X.values
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LSTM expects 3D input [samples, time_steps, features]
seq_length = 1
train_generator = TimeseriesGenerator(X_train, y_train, length=seq_length, batch_size=32)
test_generator = TimeseriesGenerator(X_test, y_test, length=seq_length, batch_size=32)

model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(seq_length, X.shape[1])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mae')

model.fit(train_generator, epochs=10, verbose=1)

# Predict
y_pred = model.predict(test_generator)
y_true = y_test[seq_length:]

# Evaluate
mae = mean_absolute_error(y_true, y_pred)
print(f"LSTM MAE: {mae:.2f}")
