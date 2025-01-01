import numpy as np

# make predictions using the trained model
def make_predictions(model, data, scaler, look_back):
    predictions = model.predict(data)
    return scaler.inverse_transform(predictions)  # convert predictions back to original scale

# forecast future values using the trained model
def forecast_future(model, data, steps=30, time_step=100):
    temp_input = data[len(data)-time_step:].reshape(1, -1).tolist()[0]  # extract the last time_step values from the data
    predictions = []
    
    for _ in range(steps):
        x_input = np.array(temp_input[-time_step:]).reshape((1, time_step, 1))  # reshape the input array's last time_step values
        prediction = model.predict(x_input, verbose=0)  # predict the next value
        predictions.append(prediction[0].tolist())
        temp_input.extend(prediction[0].tolist())
    
    return predictions