def update_weights(w, X, y_true, y_pred, lr):
    error = y_true - y_pred
    gradient = -np.dot(X.T, error) / len(y_true)  # Compute gradients
    w -= lr * gradient  # Update weights
    return w
