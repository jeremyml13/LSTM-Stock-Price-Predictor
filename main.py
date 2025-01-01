from data_processing import get_specific_stock_data, scale_data, create_dataset
from model_training import build_model, train_model
from predictions import make_predictions, forecast_future
from visualization import plot_all_predictions

# Load and preprocess data
ticker_symbol = input("Enter ticker symbol: ").strip()
start_date = input("Enter the start date (YYYY-MM-DD): ").strip()
end_date = input("Enter the end date (YYYY-MM-DD): ").strip()

stock_data = get_specific_stock_data(ticker_symbol, start_date, end_date)
scaler, scaled_data = scale_data(stock_data["Close"])

# Split data into training and testing sets
time_step = 100
training_size = int(len(scaled_data) * 0.65)
train_data, test_data = scaled_data[:training_size], scaled_data[training_size:]
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build and train the model
model = build_model((time_step, 1))
model, history = train_model(model, X_train, y_train, X_test, y_test)

# Make predictions
train_predictions = make_predictions(model, X_train, scaler, time_step)
test_predictions = make_predictions(model, X_test, scaler, time_step)

# Forecast future prices
future_predictions = forecast_future(model, test_data, steps=30, time_step=time_step)
plot_all_predictions(
    scaled_data,          # Full scaled dataset
    train_predictions,    # Predictions on training data
    test_predictions,     # Predictions on testing data
    test_data[-100:],     # Last 100 days from test data for historical view
    future_predictions,   # Future predictions for the next 30 days
    scaler,               # Scaler object to inverse-transform the data
    time_step             # Look-back period
)