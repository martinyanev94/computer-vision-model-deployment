scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update the learning rate

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, lr: {scheduler.get_last_lr()[-1]}')
