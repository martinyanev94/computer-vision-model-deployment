model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Convert 2D images to 1D
    layers.Dense(128, activation='relu'),  # Hidden layer
    layers.Dense(10, activation='softmax')  # Output layer with softmax activation
])
