def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15  # Prevent log(0) error
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping to avoid log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Example usage
y_true = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7])
bce = binary_cross_entropy(y_true, y_pred)
print(f'Binary Cross-Entropy: {bce}')
