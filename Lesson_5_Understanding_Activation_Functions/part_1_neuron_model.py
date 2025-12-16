import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        return np.dot(self.weights, inputs) + self.bias

# Example usage
weights = np.array([0.5, -0.6])
bias = 2
neuron = Neuron(weights, bias)
inputs = np.array([2, 3])
output = neuron.forward(inputs)

print("Output without activation:", output)
