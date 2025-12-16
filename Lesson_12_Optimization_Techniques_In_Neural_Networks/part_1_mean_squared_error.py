import torch

def mean_squared_error(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2)
