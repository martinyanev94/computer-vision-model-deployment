import numpy as np

def function_to_maximize(x):
    return -x**2 + 4*x

def gradient_function(x):
    return -2*x + 4

# Gradient ascent process
def gradient_ascent(starting_point, learning_rate, num_iterations):
    x = starting_point
    for _ in range(num_iterations):
        grad = gradient_function(x)
        x += learning_rate * grad
    return x

# Parameters
starting_point = 0  # where to start
learning_rate = 0.1  # how big our steps are
num_iterations = 20  # number of steps to take

max_x = gradient_ascent(starting_point, learning_rate, num_iterations)
max_value = function_to_maximize(max_x)

print(f'Maximum occurs at x = {max_x}, with a value of f(x) = {max_value}.')
