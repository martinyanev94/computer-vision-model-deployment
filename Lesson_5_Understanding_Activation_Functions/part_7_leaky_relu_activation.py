def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

# Applying Leaky ReLU activation
output_leaky_relu = leaky_relu(output)
print("Output after Leaky ReLU activation:", output_leaky_relu)
