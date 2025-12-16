import numpy as np

batch_sizes = [1, 10, 32, 64]
results = {}

for batch_size in batch_sizes:
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    losses = []

    # Creating DataLoader for batching
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(100):
        model.train()
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
        losses.append(loss.item())

    results[batch_size] = losses

# Visualization
for batch_size, losses in results.items():
    plt.plot(losses, label=f'Batch Size: {batch_size}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Effect of Batch Size on Training Loss')
plt.legend()
plt.show()
