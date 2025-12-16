class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def activate(self, inputs):
        total_input = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(total_input)
