from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout

model = Sequential()  # A sequential model

# Adding layers to the model
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))  # Input layer
model.add(Flatten())  # Flatten the 2D output to 1D
model.add(Dense(128, activation='relu'))  # Hidden layer
model.add(Dropout(0.5))  # Dropout layer to reduce overfitting
model.add(Dense(10, activation='softmax'))  # Output layer for classification

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # Compile the model
