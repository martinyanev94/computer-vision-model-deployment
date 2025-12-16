def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -alpha * (1 - y_pred) ** gamma * y_true * np.log(y_pred)
    return np.sum(loss) / len(y_true)

# Example usage with a single sample
y_true = np.array([[1, 0, 0]])
y_pred = np.array([[0.9, 0.05, 0.05]])
fl = focal_loss(y_true, y_pred)
print(f"Focal Loss: {fl}")
