import pandas as pd
import tensorflow as tf
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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')
history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_data=(X_validation_scaled, y_validation))

loss = model.evaluate(X_test_scaled, y_test)
predictions = model.predict(X_test_scaled)

model.save('best_model_with_regularization.h5')
