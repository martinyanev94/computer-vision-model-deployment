def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * np.square(error)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss).mean()

# Example usage
y_true = np.array([2, 1, 3, 4])
y_pred = np.array([1.5, 1.2, 3.5, 5])
huber = huber_loss(y_true, y_pred)
print(f'Huber Loss: {huber}')
