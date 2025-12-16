from torch.optim.lr_scheduler import StepLR

optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    # Training routine

    optimizer.step()  # Update weights
    scheduler.step()  # Update learning rate
