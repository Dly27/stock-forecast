import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries

ALPHA_VANTAGE_API_KEY = 'VC2S9T9RSVXMPOP0'

def fetch_stock_data(symbol, start_date, end_date):
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
    data = data[(data.index >= start_date) & (data.index <= end_date)]
    return data

def prepare_data(data):
    features = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
    target = data['4. close']
    return features, target

stock_data = fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
features, target = prepare_data(stock_data)

X_train, X_temp, y_train, y_temp = train_test_split(features, target, test_size=0.3, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

pipeline = Pipeline([
    ('scaler', MinMaxScaler())
])

datasets = [X_train, X_validation, X_test]
scaled_datasets = []
for dataset in datasets:
    scaled_dataset = pipeline.fit_transform(dataset)
    scaled_datasets.append(scaled_dataset)

X_train_scaled, X_validation_scaled, X_test_scaled = scaled_datasets

time_steps = 30
X_train_lstm = []
y_train_lstm = []

for i in range(time_steps, len(X_train_scaled)):
    X_train_lstm.append(X_train_scaled[i - time_steps:i])
    y_train_lstm.append(y_train[i])

X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)
X_train_lstm = X_train_lstm.reshape(X_train_lstm.shape[0], X_train_lstm.shape[1], X_train_lstm.shape[2])

X_validation_lstm = []
y_validation_lstm = []

for i in range(time_steps, len(X_validation_scaled)):
    X_validation_lstm.append(X_validation_scaled[i - time_steps:i])
    y_validation_lstm.append(y_validation[i])

X_validation_lstm, y_validation_lstm = np.array(X_validation_lstm), np.array(y_validation_lstm)
X_validation_lstm = X_validation_lstm.reshape(X_validation_lstm.shape[0], X_validation_lstm.shape[1], X_validation_lstm.shape[2])

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=100,
    batch_size=32,
    validation_data=(X_validation_lstm, y_validation_lstm),
    callbacks=[early_stopping]
)

X_test_lstm = []
y_test_lstm = []

for i in range(time_steps, len(X_test_scaled)):
    X_test_lstm.append(X_test_scaled[i - time_steps:i])
    y_test_lstm.append(y_test[i])

X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)
X_test_lstm = X_test_lstm.reshape(X_test_lstm.shape[0], X_test_lstm.shape[1], X_test_lstm.shape[2])

loss = model.evaluate(X_test_lstm, y_test_lstm)
predictions = model.predict(X_test_lstm)

model.save('best_model_with_lstm.keras')
