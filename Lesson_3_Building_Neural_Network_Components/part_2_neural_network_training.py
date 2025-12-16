def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def backpropagation(self, X, y, learning_rate):
    output_errors = y - self.output_layer_activation
    hidden_errors = np.dot(output_errors, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_layer_output)

    # Update weights
    self.weights_hidden_output += np.dot(self.hidden_layer_output.T, output_errors) * learning_rate
    self.weights_input_hidden += np.dot(X.T, hidden_errors) * learning_rate

def relu_derivative(self, x):
    return np.where(x > 0, 1, 0)
