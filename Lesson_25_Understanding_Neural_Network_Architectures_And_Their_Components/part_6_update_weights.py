def update_weights(self, input_data, hidden_output, output, y_true, learning_rate):
    # Calculate the error at the output layer
    output_error = output - y_true
    output_delta = output_error * output * (1 - output)  # Derivative of sigmoid

    # Calculate the error at the hidden layer
    hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
    hidden_delta = hidden_error * hidden_output * (1 - hidden_output)  # Derivative of sigmoid

    # Update weights for the output layer
    self.weights_hidden_output -= learning_rate * np.dot(hidden_output.T.reshape(-1, 1), output_delta.reshape(1, -1))

    # Update weights for the input-hidden layer
    self.weights_input_hidden -= learning_rate * np.dot(input_data.reshape(-1, 1), hidden_delta.reshape(1, -1))
