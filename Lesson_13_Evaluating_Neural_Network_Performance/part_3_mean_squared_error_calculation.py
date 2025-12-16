# Define the Mean Squared Error loss function
def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Example output from our feedforward function and true labels
y_true = np.array([[0], [1], [1], [0]])  # True binary labels
loss = mean_squared_error(y_true, output)
print("Mean Squared Error Loss:", loss)
