def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Actual output
y_true = np.array([0.7])
mse = mean_squared_error(y_true, output)
print(f"Mean Squared Error: {mse}")
