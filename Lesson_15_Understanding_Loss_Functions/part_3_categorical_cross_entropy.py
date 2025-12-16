def categorical_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clipping
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

# Example usage
y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  # One-hot encoding
y_pred = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6]])
cce = categorical_cross_entropy(y_true, y_pred)
print(f'Categorical Cross-Entropy: {cce}')
