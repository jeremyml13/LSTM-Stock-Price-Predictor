import matplotlib.pyplot as plt
import numpy as np

# plot both of the graphs - historical data predictions and future predictions - in one display
def plot_all_predictions(data, train_predict, test_predict, last_100_days, future_predictions, scaler, look_back):
    
    train_plot = np.empty_like(data)
    train_plot[:, :] = np.nan
    train_plot[look_back:len(train_predict)+look_back, :] = train_predict

    test_plot = np.empty_like(data)
    test_plot[:, :] = np.nan
    test_plot[len(train_predict)+(look_back*2)+1:len(data)-1, :] = test_predict

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))

    # Plot historical data with train/test predictions
    axes[0].plot(scaler.inverse_transform(data), label="Original Data")
    axes[0].plot(train_plot, label="Train Predictions")
    axes[0].plot(test_plot, label="Test Predictions")
    axes[0].set_title("Historical Data with Train/Test Predictions")
    axes[0].legend()

    # Plot future predictions
    days = np.arange(len(last_100_days) + len(future_predictions))
    axes[1].plot(days[:len(last_100_days)], scaler.inverse_transform(last_100_days), label="Historical Data")
    axes[1].plot(days[len(last_100_days):], scaler.inverse_transform(future_predictions), label="Future Predictions")
    axes[1].set_title("Future Predictions")
    axes[1].legend()

    plt.tight_layout()
    plt.show()