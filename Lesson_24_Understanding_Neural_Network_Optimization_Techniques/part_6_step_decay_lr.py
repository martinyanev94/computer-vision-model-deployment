def step_decay(epoch, initial_lr=0.1, drop_epoch=10, drop_factor=0.5):
    if epoch % drop_epoch == 0 and epoch > 0:
        return initial_lr * drop_factor
    return initial_lr
