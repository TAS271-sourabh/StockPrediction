import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time
import os

device = torch.device('cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

def load_trained_model(stock_symbol):
    model_path = f'stock_price_model_{stock_symbol}.pth'  # Adjust if needed
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model for {stock_symbol} not found at {model_path}")

    model = LSTMModel(input_size=4, hidden_size=50, output_size=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_price(stock_symbol, date_str):
    start_time = time.time()
    
    # Load the model for the corresponding stock symbol
    model = load_trained_model(stock_symbol)

    # Load the dataset corresponding to the stock symbol
    data_file_path = f'../Data/{stock_symbol}_data.csv'  # Dynamic data loading based on stock symbol
    if not os.path.exists(data_file_path):
        return {"error": f"Data file for stock {stock_symbol} not found."}
    
    data = pd.read_csv(data_file_path, index_col='Date', parse_dates=True)
    date = pd.to_datetime(date_str)

    # Use the last 30 days of past data before the given date
    past_data = data[data.index < date].tail(30)
    if len(past_data) < 30:
        return {"error": f"Not enough data to predict for {date_str}."}

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Open', 'Close', 'High', 'Low']])
    past_data_scaled = scaler.transform(past_data[['Open', 'Close', 'High', 'Low']])
    X_input = torch.tensor(past_data_scaled, dtype=torch.float32).view(1, 30, -1).to(device)

    # Make predictions
    with torch.no_grad():
        predicted = model(X_input)

    predicted_prices = predicted.cpu().numpy()[0]

    # Prepare for inverse scaling
    dummy_array = np.zeros((1, 4))
    dummy_array[0, 0] = predicted_prices[0]  # Predicted open price
    dummy_array[0, 1] = predicted_prices[1]  # Predicted close price
    inverse_prices = scaler.inverse_transform(dummy_array)

    try:
        # Get the actual open and close prices for the given date
        actual_price = data.loc[date_str]
        actual_open = actual_price['Open']
        actual_close = actual_price['Close']

        # Calculate accuracy
        open_accuracy = 100 - (abs(inverse_prices[0][0] - actual_open) / actual_open * 100)
        close_accuracy = 100 - (abs(inverse_prices[0][1] - actual_close) / actual_close * 100)

        end_time = time.time()
        execution_time = end_time - start_time

        return {
            "stockSymbol": stock_symbol,
            "date": date_str,
            "actual_open": actual_open,
            "actual_close": actual_close,
            "predicted_open": inverse_prices[0][0],
            "predicted_close": inverse_prices[0][1],
            "open_accuracy": open_accuracy,
            "close_accuracy": close_accuracy,
            "execution_time": execution_time
        }
    except KeyError:
        return {"error": f"No actual prices available for the date: {date_str}."}
