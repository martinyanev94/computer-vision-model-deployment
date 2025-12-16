class MomentumOptimizer:
    def __init__(self, model_params, lr=0.01, momentum=0.9):
        self.params = list(model_params)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(param) for param in self.params]

    def step(self):
        for i, param in enumerate(self.params):
            self.velocities[i] = self.momentum * self.velocities[i] + self.lr * param.grad.data
            param.data -= self.velocities[i]
