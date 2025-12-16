# Define the backpropagation function
def backpropagation(X, y_true, y_pred, weights, learning_rate):
    # Calculating the derivatives using the chain rule
    output_error = y_pred - y_true  # Error at output
    layer2_output = relu(np.dot(relu(np.dot(X, weights[0])), weights[1])) 

    # Derivatives for output layer
    output_gradient = output_error * 1  # Since it's linear at output 
    weights[2] -= learning_rate * np.dot(layer2_output.T, output_gradient)

    # Proceeding backwards for the hidden layers
    layer2_error = np.dot(output_gradient, weights[2].T) * (layer2_output > 0)  # ReLU derivative
    weights[1] -= learning_rate * np.dot(relu(np.dot(X, weights[0])).T, layer2_error)

    layer1_error = np.dot(layer2_error, weights[1].T) * (X > 0)  # ReLU derivative
    weights[0] -= learning_rate * np.dot(X.T, layer1_error)

# Hyperparameters
learning_rate = 0.01

# Run backpropagation
backpropagation(X, y_true, output, weights, learning_rate)
print("Weights updated successfully.")
