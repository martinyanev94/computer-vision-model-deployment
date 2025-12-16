from torch.optim.lr_scheduler import StepLR

# Assume optimizer is already defined
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(epochs):
    train(model, criterion, optimizer, data_loader, 1)  # Train for 1 epoch
    scheduler.step()  # Update the learning rate
