import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)

    def activate(self, x):
        # Using the sigmoid function as an activation function
        return 1 / (1 + np.exp(-x))

    def feedforward(self, input_data):
        hidden_layer_input = np.dot(input_data, self.weights_input_hidden)
        hidden_layer_output = self.activate(hidden_layer_input)

        output_layer_input = np.dot(hidden_layer_output, self.weights_hidden_output)
        output_layer_output = self.activate(output_layer_input)

        return output_layer_output
