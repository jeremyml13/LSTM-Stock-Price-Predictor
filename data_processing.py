import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Fetch stock data for a specified ticker symbol
def get_specific_stock_data(ticker_symbol, start_date, end_date):
    
    ticker_data = yf.Ticker(ticker_symbol)
    stock_data = ticker_data.history(period='1d', start=start_date, end=end_date).reset_index()  # stock data for inputted ticker
    
    # Add additional date-related features
    stock_data['DayOfYear'] = stock_data['Date'].dt.dayofyear
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['DayOfWeek'] = stock_data['Date'].dt.dayofweek
    stock_data["DayChange"] = stock_data["Open"] - stock_data["Close"]
    stock_data["DayChangeIndicator"] = stock_data["DayChange"].apply(lambda x: 1 if x > 0 else 0)  # postive or negative indicator (1 is positive)

    stock_data = stock_data.sort_values(by='Date', ascending=True).reset_index(drop=True)
    
    return stock_data

# Scale data to a range of [0, 1]
def scale_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    return scaler, scaler.fit_transform(np.array(data).reshape(-1, 1))

# Create a dataset for time series prediction
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        dataX.append(dataset[i:(i+time_step), 0])  # append input features
        dataY.append(dataset[i + time_step, 0])  # append target value
    return np.array(dataX), np.array(dataY)