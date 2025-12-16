def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Applying Sigmoid activation
output_sigmoid = sigmoid(output)
print("Output after Sigmoid activation:", output_sigmoid)
