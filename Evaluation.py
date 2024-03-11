import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from alpha_vantage.timeseries import TimeSeries
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

class ModelEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.loaded_model = tf.keras.models.load_model(model_path)

    def fetch_stock_data(self, symbol, start_date, end_date):
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=symbol, outputsize='full')
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        return data

    def prepare_data(self, data):
        features = data[['1. open', '2. high', '3. low', '4. close', '5. volume']]
        target = data['4. close']
        return features, target

    def evaluate_model(self, X_test_scaled, y_test):
        predictions = self.loaded_model.predict(X_test_scaled).flatten()
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return mse, mae, rmse, predictions

    def plot_actual_vs_predicted(self, y_test, predictions):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, predictions, color='blue', label='Actual vs Predicted')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=3, color='red')
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted Prices')
        plt.legend()
        plt.show()

    def plot_residuals(self, y_test, predictions):
        residuals = y_test - predictions
        plt.figure(figsize=(10, 6))
        plt.scatter(predictions, residuals, color='blue')
        plt.axhline(y=0, color='red', linestyle='--', lw=2)
        plt.xlabel('Predicted Price')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

    def plot_residuals_density(self, residuals):
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, color='blue')
        plt.xlabel('Residuals')
        plt.ylabel('Density')
        plt.title('Density Plot of Residuals')
        plt.show()

    def plot_actual_vs_predicted_time(self, y_test, predictions):
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label='Actual Price', color='blue')
        plt.plot(y_test.index, predictions, label='Predicted Price', color='red')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Actual vs Predicted Prices over Time')
        plt.legend()
        plt.show()

    def plot_feature_importance(self):
        feature_importance = self.loaded_model.get_weights()[0]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.show()

ALPHA_VANTAGE_API_KEY = 'VC2S9T9RSVXMPOP0'

model_evaluator = ModelEvaluator('sequential_model.h5')

stock_data = model_evaluator.fetch_stock_data('AAPL', '2020-01-01', '2024-01-01')
features, target = model_evaluator.prepare_data(stock_data)

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

mse, mae, rmse, predictions = model_evaluator.evaluate_model(X_test_scaled, y_test)
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

model_evaluator.plot_actual_vs_predicted(y_test, predictions)
model_evaluator.plot_residuals(y_test, predictions)
residuals = y_test - predictions
model_evaluator.plot_residuals_density(residuals)
model_evaluator.plot_actual_vs_predicted_time(y_test, predictions)
model_evaluator.plot_feature_importance()
