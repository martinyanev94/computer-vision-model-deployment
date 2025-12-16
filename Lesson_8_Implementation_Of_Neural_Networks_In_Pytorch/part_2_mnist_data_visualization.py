transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
def visualize_data(loader):
    samples = next(iter(loader))
    images, labels = samples
    grid_img = torchvision.utils.make_grid(images, nrow=8, padding=1)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0).numpy())
    plt.title("Sample MNIST Images")
    plt.axis('off')
    plt.show()

visualize_data(train_loader)
