import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([0.5, 0.8, 0.2])
y_pred = np.array([0.4, 0.7, 0.3])
loss = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {loss}")
