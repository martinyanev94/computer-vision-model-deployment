import numpy as np

class FeedForwardNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.hidden_layer_output = None

    def feedforward(self, input_data):
        self.hidden_layer_output = self.sigmoid(np.dot(input_data, self.weights_input_hidden))
        output = self.sigmoid(np.dot(self.hidden_layer_output, self.weights_hidden_output))
        return output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
