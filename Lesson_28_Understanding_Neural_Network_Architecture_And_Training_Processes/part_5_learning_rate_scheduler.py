# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

for epoch in range(100):
    optimizer.zero_grad()
    inputs = torch.rand(1, input_size)
    labels = torch.tensor([1])
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    scheduler.step()  # This adjusts the learning rate based on the schedule
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/100], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]}')
