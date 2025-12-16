learning_rates = [0.01, 0.1, 0.001]  # Different learning rates
results = {}

for lr in learning_rates:
    model = SimpleNN()  # Reinitialize the model for each learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr)
    losses = []  # To store loss values for each epoch
    
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    results[lr] = losses

# Visualization of the training loss
import matplotlib.pyplot as plt

for lr, losses in results.items():
    plt.plot(losses, label=f'Learning Rate: {lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Effect of Learning Rate on Training Loss')
plt.legend()
plt.show()
