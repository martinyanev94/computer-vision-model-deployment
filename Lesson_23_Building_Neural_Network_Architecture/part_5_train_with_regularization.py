def train_with_regularization(self, inputs, targets, epochs=10000, l2_lambda=0.01):
        for epoch in range(epochs):
            hidden_input = np.dot(inputs, self.weights_input_hidden)
            hidden_output = self.sigmoid(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)

            error = targets - final_output
            d_final_output = error * self.sigmoid_derivative(final_output)

            # Adding L2 penalty to the output layer
            self.weights_hidden_output += hidden_output.T.dot(d_final_output) * self.learning_rate
            self.weights_hidden_output -= l2_lambda * self.weights_hidden_output

            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_output)

            # Updating weights for input to hidden layer
            self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * self.learning_rate
            self.weights_input_hidden -= l2_lambda * self.weights_input_hidden

# Usage of the regularization training method
ann.train_with_regularization(training_data, target_data)
