def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Example actual labels for our sample data
y_true = np.array([[0], [1], [1], [0]])
loss = mean_squared_error(y_true, output)
print("Mean Squared Error Loss:", loss)
