def backpropagation(X, y_true, y_pred, weights, learning_rate):
    output_error = y_pred - y_true  # Calculate error at output layer
    layer2_output = relu(np.dot(relu(np.dot(X, weights[0])), weights[1]))

    # Calculate gradients for weights
    output_gradient = output_error * 1  # Gradient for output layer
    weights[2] -= learning_rate * np.dot(layer2_output.T, output_gradient)  # Update output layer weights

    layer2_error = np.dot(output_gradient, weights[2].T) * (layer2_output > 0)  # ReLU derivative
    weights[1] -= learning_rate * np.dot(relu(np.dot(X, weights[0])).T, layer2_error)  # Update second hidden layer weights

    layer1_error = np.dot(layer2_error, weights[1].T) * (X > 0)  # ReLU derivative
    weights[0] -= learning_rate * np.dot(X.T, layer1_error)  # Update first hidden layer weights

# Example usage
learning_rate = 0.01
backpropagation(X, y_true, output, weights, learning_rate)
print("Weights updated successfully.")
