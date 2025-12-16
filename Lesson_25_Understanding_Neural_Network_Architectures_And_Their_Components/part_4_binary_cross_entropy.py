def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-12) + (1 - y_true) * np.log(1 - y_pred + 1e-12))
