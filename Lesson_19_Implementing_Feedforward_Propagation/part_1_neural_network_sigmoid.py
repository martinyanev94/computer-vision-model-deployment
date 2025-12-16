import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid to use in backpropagation
def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize the weights randomly with mean 0
np.random.seed(1)
input_layer_size = 2
hidden_layer_size = 2
output_layer_size = 1

# Weights for input to hidden layer
weights_input_hidden = np.random.rand(input_layer_size, hidden_layer_size)
# Weights for hidden to output layer
weights_hidden_output = np.random.rand(hidden_layer_size, output_layer_size)
