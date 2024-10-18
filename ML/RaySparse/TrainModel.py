import os
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time
import datetime

def train_stock_model(stock_symbol):
    # Define the directory to save the data
    data_dir = os.path.join(os.path.dirname(__file__), '../Data')
    os.makedirs(data_dir, exist_ok=True)  # Create Data directory if it doesn't exist

    # Calculate the start and end dates
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=10 * 365)  # 10 years

    # Fetch stock data using start and end dates
    stock_data = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d")

    # Check if data is fetched properly
    if stock_data.empty:
        return {"error": f"No stock data found for symbol: {stock_symbol}"}

    # Save the stock data to CSV in the Data directory
    csv_file_path = os.path.join(data_dir, f"{stock_symbol}_data.csv")
    stock_data.to_csv(csv_file_path)

    # Load the data from the CSV file for training
    stock_data = pd.read_csv(csv_file_path)

    # Preprocess the data
    stock_data = stock_data[['Open', 'Close', 'High', 'Low']].dropna()  # Ensure no NaN values

    # Check if sufficient data is available
    if stock_data.shape[0] == 0:
        return {"error": f"Not enough data for stock symbol: {stock_symbol}"}

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data)

    # Create sequences
    def create_sequences(data, seq_length=30):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length][[0, 1]])  # Predict Open and Close
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data)

    # Check if sequences are created
    if X.shape[0] == 0 or y.shape[0] == 0:
        return {"error": f"Insufficient data after processing for stock symbol: {stock_symbol}"}

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the LSTM model
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])  # Use the last time step

    # Model parameters
    input_size = X.shape[2]
    hidden_size = 50
    output_size = 2  # Open and Close prices

    model = LSTMModel(input_size, hidden_size, output_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start = time.time()  # Start timing the training

    # Training the model
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.FloatTensor(X_train))
        loss = criterion(outputs, torch.FloatTensor(y_train))
        loss.backward()
        optimizer.step()
    
    end = time.time()  # End timing the training

    training_time = end - start
    
    # Save the trained model
    model_path = f'stock_price_model_{stock_symbol}.pth'
    torch.save(model.state_dict(), model_path)

    return {
        "message": f"Model trained and saved as {model_path} for stock {stock_symbol}",
        "training_time": training_time
    }
