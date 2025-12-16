# Stochastic Gradient Descent loop
for epoch in range(100):
    for i in range(len(x)):
        optimizer.zero_grad()  # Zero the gradients
        x_i = x[i].unsqueeze(0)  # Grab the single sample
        y_i = y[i].unsqueeze(0)
        
        y_pred = model(x_i)  # Forward pass
        loss = criterion(y_pred, y_i)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
