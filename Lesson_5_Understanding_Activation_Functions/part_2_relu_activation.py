def relu(x):
    return np.maximum(0, x)

# Applying ReLU activation
output_relu = relu(output)
print("Output after ReLU activation:", output_relu)
