import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Sample model
model = keras.Sequential([
    layers.Dense(32, activation='relu', input_shape=(input_dimension,)),
    layers.Dense(3, activation='softmax')  # Assuming 3 classes for classification
])

# Compile the model with categorical crossentropy loss
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
