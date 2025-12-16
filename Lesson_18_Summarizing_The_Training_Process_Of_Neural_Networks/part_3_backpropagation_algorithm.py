def backpropagation(self, x, y_true):
    output_loss = self.final_output - y_true
    d_weights_hidden_output = np.dot(self.hidden_layer_output.T, output_loss)
    
    # For the hidden layer
    hidden_layer_gradient = output_loss.dot(self.weights_hidden_output.T) * self.hidden_layer_output * (1 - self.hidden_layer_output)
    d_weights_input_hidden = np.dot(x.reshape(-1, 1), hidden_layer_gradient)

    # Update weights
    self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
    self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden

# After calculating the forward pass and the loss, invoking backpropagation:
nn.backpropagation(input_data, y_true)
