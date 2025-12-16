import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        self.weights_input_hidden = np.random.rand(input_size, hidden_size)
        self.weights_hidden_output = np.random.rand(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        self.hidden_layer_activation = np.dot(x, self.weights_input_hidden)
        self.hidden_layer_output = self.activation_function(self.hidden_layer_activation)
        self.final_output = np.dot(self.hidden_layer_output, self.weights_hidden_output)
        return self.final_output

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))  # Sigmoid activation

# Example usage: instantiate the neural network with 3 input features, 2 hidden neurons, and 1 output
nn = SimpleNeuralNetwork(input_size=3, hidden_size=2, output_size=1, learning_rate=0.01)

# Forward pass with dummy input data
input_data = np.array([0.5, 0.1, 0.4])
output = nn.forward(input_data)
print(f"Output from forward propagation: {output}")
