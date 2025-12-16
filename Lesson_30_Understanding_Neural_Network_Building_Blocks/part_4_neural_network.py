class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_layer = Neuron(np.random.rand(input_size, hidden_size), np.random.rand(hidden_size))
        self.output_layer = Neuron(np.random.rand(hidden_size, output_size), np.random.rand(output_size))

    def forward(self, inputs):
        hidden_output = self.hidden_layer.activate(inputs)
        output = self.output_layer.activate(hidden_output)
        return output

    def train(self, inputs, expected_output, learning_rate):
        output = self.forward(inputs)
        loss = np.mean((expected_output - output) ** 2)

        # Calculate gradients (This is a simplified example)
        output_gradient = -2 * (expected_output - output)
        hidden_output = self.hidden_layer.activate(inputs)
        
        # Update weights (simplified)
        self.output_layer.weights -= learning_rate * hidden_output * output_gradient
        self.hidden_layer.weights -= learning_rate * inputs * output_gradient.dot(self.output_layer.weights.T)

        return loss
