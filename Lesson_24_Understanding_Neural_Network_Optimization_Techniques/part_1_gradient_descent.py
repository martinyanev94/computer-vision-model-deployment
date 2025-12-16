import numpy as np

def gradient_descent(weights, learning_rate, gradients):
    return weights - learning_rate * gradients
