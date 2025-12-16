def l2_regularization(weights, lambda_reg):
    return lambda_reg * np.sum(np.square(weights))
