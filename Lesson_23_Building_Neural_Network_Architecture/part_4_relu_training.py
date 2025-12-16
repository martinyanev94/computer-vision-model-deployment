def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def train_with_relu(self, inputs, targets, epochs=10000):
        for epoch in range(epochs):
            hidden_input = np.dot(inputs, self.weights_input_hidden)
            hidden_output = self.relu(hidden_input)

            final_input = np.dot(hidden_output, self.weights_hidden_output)
            final_output = self.sigmoid(final_input)  # keeping output layer's sigmoid for binary classification

            error = targets - final_output

            d_final_output = error * self.sigmoid_derivative(final_output)
            error_hidden_layer = d_final_output.dot(self.weights_hidden_output.T)
            d_hidden_layer = error_hidden_layer * self.relu_derivative(hidden_output)

            self.weights_hidden_output += hidden_output.T.dot(d_final_output) * self.learning_rate
            self.weights_input_hidden += inputs.T.dot(d_hidden_layer) * self.learning_rate

# Using the ReLU trained method
ann.train_with_relu(training_data, target_data)
