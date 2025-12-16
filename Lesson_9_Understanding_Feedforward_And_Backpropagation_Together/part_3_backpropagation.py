def backpropagation(self, input_data, target_data, learning_rate):
    # Perform a feedforward pass
    predictions = self.feedforward(input_data)
    
    # Calculate the error
    output_layer_error = predictions - target_data
    hidden_layer_error = np.dot(output_layer_error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_layer_output)

    # Update weights
    self.weights_hidden_output -= learning_rate * np.dot(self.hidden_layer_output.T, output_layer_error)
    self.weights_input_hidden -= learning_rate * np.dot(input_data.T, hidden_layer_error)

@staticmethod
def sigmoid_derivative(output):
    return output * (1 - output)
