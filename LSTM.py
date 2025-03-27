import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# ---------------------------
# 1. Data Reading and Preprocessing
# ---------------------------
file_path = r'C:\Users\于嘉诚\Desktop\STDM\beijing_daily_avg_aqi_all_years.csv'
df = pd.read_csv(file_path)

# Convert the 'Date' column to datetime format and sort by date
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
df = df.sort_values(by='Date')

# Check for missing values and fill them using the previous day's data (forward fill)
if df.isnull().values.any():
    print("Missing values detected, filling with previous day's data.")
    df = df.fillna(method='ffill')

# Set the target forecast date: 2025/06/01
forecast_date = pd.to_datetime("2025-06-01")
last_date = df['Date'].max()  # The last date in the dataset (e.g., 2025-03-01)
forecast_steps = (forecast_date - last_date).days  # Number of days to forecast
print(
    f"From {last_date.date()} to {forecast_date.date()}, need to forecast {forecast_steps} days of data.\n")


# ---------------------------
# 2. Define Time Series Data Conversion Function
# ---------------------------
def create_dataset(dataset, look_back=1):
    """
    Convert a one-dimensional time series into a supervised learning dataset.
    Parameters:
      dataset: time series data as a numpy array.
      look_back: number of time steps to use for prediction.
    Returns:
      X: input data, shape [number of samples, look_back].
      Y: output data, shape [number of samples, ].
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back)])
        Y.append(dataset[i + look_back])
    return np.array(X), np.array(Y)


# ---------------------------
# 3. Set Hyperparameters
# ---------------------------
look_back = 30  # Use the past 30 days of data to predict the next day
train_ratio = 0.9  # First 90% for training, last 10% for testing
epochs = 50  # Number of training epochs
batch_size = 32  # Batch size for training

# ---------------------------
# 4. Train, Evaluate, and Forecast Future Data for Each Station
# ---------------------------
# Assume the first column is 'Date' and the remaining columns are the data for each station
station_names = df.columns[1:]
rmse_results = {}  # RMSE for each station's test set
test_predictions_list = []  # List to store test set predictions (after inverse normalization) for each station
test_true_list = []  # List to store true test set values (after inverse normalization) for each station
forecast_results = {}  # Future forecast results for each station

for station in station_names:
    print(f"Station: {station}")
    # Extract data for the current station and reshape to 2D array
    data = df[station].values.reshape(-1, 1)

    # Normalize data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Construct supervised learning dataset using sliding window
    X, Y = create_dataset(data_scaled, look_back=look_back)

    # Split data into training and test sets (chronologically)
    train_size = int(len(X) * train_ratio)
    X_train, Y_train = X[:train_size], Y[:train_size]
    X_test, Y_test = X[train_size:], Y[train_size:]

    # Reshape data for LSTM input: [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # ---------------------------
    # 5. Build and Train the LSTM Model
    # ---------------------------
    model = Sequential()
    model.add(LSTM(32, input_shape=(look_back, 1)))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # ---------------------------
    # 6. Predict on the Test Set and Compute RMSE (Single Station)
    # ---------------------------
    predictions = model.predict(X_test)
    predictions_inverted = scaler.inverse_transform(predictions)
    Y_test_inverted = scaler.inverse_transform(Y_test.reshape(-1, 1))

    station_rmse = math.sqrt(
        mean_squared_error(Y_test_inverted, predictions_inverted))
    rmse_results[station] = station_rmse
    print(f"Test Set RMSE: {station_rmse:.4f}")

    # Store predictions and true values for averaging later
    test_predictions_list.append(predictions_inverted.flatten())
    test_true_list.append(Y_test_inverted.flatten())

    # ---------------------------
    # 7. Forecast Future Data up to the Target Date Using the Model
    # ---------------------------
    last_window = data_scaled[
                  -look_back:].copy()  # initial window (shape: (look_back, 1))
    forecast_sequence = []
    for _ in range(forecast_steps):
        pred_step = model.predict(last_window.reshape(1, look_back, 1))
        forecast_sequence.append(pred_step[0, 0])
        # Update window: remove earliest value, add latest prediction
        last_window = np.append(last_window[1:], pred_step, axis=0)

    forecast_array = np.array(forecast_sequence).reshape(-1, 1)
    forecast_array_inverted = scaler.inverse_transform(forecast_array).flatten()
    forecast_results[station] = forecast_array_inverted
    print(
        f"Forecast data from {(last_date + timedelta(days=1)).date()} to {forecast_date.date()} has been generated.\n")

# ---------------------------
# 8. Output RMSE Results for Each Station and Overall Evaluation Metrics
# ---------------------------
print("======== Test Set RMSE for Each Station ========")
for station, rmse_val in rmse_results.items():
    print(f"{station}: RMSE = {rmse_val:.4f}")

# Convert lists to numpy arrays for averaging (shape: [num_stations, num_samples])
test_predictions_array = np.array(test_predictions_list)
test_true_array = np.array(test_true_list)

# Compute overall average (across stations) for each test sample
avg_test_predictions = np.mean(test_predictions_array, axis=0)
avg_test_true = np.mean(test_true_array, axis=0)

overall_rmse = math.sqrt(
    mean_squared_error(avg_test_true, avg_test_predictions))
overall_r2 = r2_score(avg_test_true, avg_test_predictions)
overall_mae = mean_absolute_error(avg_test_true, avg_test_predictions)

print(f"\nOverall Test Set RMSE = {overall_rmse:.4f}")
print(f"Overall Test Set R2   = {overall_r2:.4f}")
print(f"Overall Test Set MAE  = {overall_mae:.4f}")

# ---------------------------
# 9. Organize Future Forecast Results into a DataFrame and Save to Excel
# ---------------------------
# Construct a sequence of forecast dates: from the day after the last date to forecast_date (inclusive)
forecast_dates = pd.date_range(start=last_date + timedelta(days=1),
                               end=forecast_date, freq='D')

forecast_df = pd.DataFrame(
    {station: forecast_results[station] for station in station_names},
    index=forecast_dates)
forecast_df.index.name = "Date"
forecast_df.reset_index(inplace=True)

cols = ["Date"] + list(station_names)
forecast_df = forecast_df[cols]

# ---------------------------
# 10. Append a Row of Overall Evaluation Metrics to the End of the DataFrame
# ---------------------------
metrics_row = {"Date": "Evaluation Metrics"}
if len(station_names) >= 3:
    metrics_row[station_names[0]] = f"Overall RMSE: {overall_rmse:.4f}"
    metrics_row[station_names[1]] = f"Overall R2: {overall_r2:.4f}"
    metrics_row[station_names[2]] = f"Overall MAE: {overall_mae:.4f}"
for station in station_names[3:]:
    metrics_row[station] = ""

metrics_df = pd.DataFrame([metrics_row])
forecast_df = pd.concat([forecast_df, metrics_df], ignore_index=True)

output_path = r'C:\Users\于嘉诚\Desktop\STDM\LSTM_Predict.xlsx'
forecast_df.to_excel(output_path, index=False)
print(
    f"\nForecast results and overall evaluation metrics have been saved to the Excel file: {output_path}")

# ---------------------------
# 11. Plotting Average Actual vs. Predicted AQI on the Test Set
# ---------------------------
# Compute test set dates based on the first station's supervised dataset
total_samples = len(df) - look_back
train_size = int(total_samples * train_ratio)
# Test dates: from index (look_back + train_size) 到末尾
test_dates = df['Date'].iloc[look_back + train_size:].reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(test_dates, avg_test_true, label='Actual', linestyle='-',
         color='black')
plt.plot(test_dates, avg_test_predictions, label='Predicted', linestyle='--',
         color='black')
plt.xlabel('Time')
plt.ylabel('AQI')
plt.legend()

save_path = r'C:\Users\于嘉诚\Desktop\STDM\Fig_LSTM.png'
plt.savefig(save_path)
plt.close()
print(f"\nThe plot has been saved to: {save_path}")
