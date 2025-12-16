import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Example usage
y_true = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])
mse = mean_squared_error(y_true, y_pred)
print(f"Mean Squared Error: {mse}")
