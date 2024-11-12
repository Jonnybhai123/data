# Import necessary packages
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Conv2D, Dense, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import mnist
# Load and preprocess the image data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape) # Output shape of training data
# Normalize the images to the range [0, 1]
X_train = (X_train.astype('float32') / 255.0)
X_test = (X_test.astype('float32') / 255.0)
# Function to plot digits
def plot_digit(image, digit, plt, i):
 plt.subplot(4, 5, i + 1)
 plt.imshow(image, cmap='gray')
 plt.title(f"Digit: {digit}")
 plt.xticks([])
 plt.yticks([])
# Plot the first 20 digits
plt.figure(figsize=(16, 10))
for i in range(20):
 plot_digit(X_train[i], y_train[i], plt, i)
plt.show()
# Reshape the data for the CNN
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
# Define the model’s architecture
model = Sequential([
 Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
 MaxPooling2D((2, 2)),
 Flatten(),
 Dense(100, activation="relu"),
 Dense(10, activation="softmax")
])
# Compile the model
optimizer = SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer=optimizer,
 loss="sparse_categorical_crossentropy",
 metrics=["accuracy"])
model.summary()
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Plot predictions on random test images
plt.figure(figsize=(16, 10))
for i in range(20):
 image = random.choice(X_test).squeeze()
 digit = np.argmax(model.predict(image.reshape((1, 28, 28, 1))), axis=-1)
 plot_digit(image, digit, plt, i)
plt.show()
# Estimate the model’s performance
predictions = np.argmax(model.predict(X_test), axis=-1)
accuracy = accuracy_score(y_test, predictions)
print(f'Test accuracy: {accuracy * 100:.2f}%')
# Display a random test image and its prediction
n = random.randint(0, len(X_test) - 1)
plt.imshow(X_test[n].squeeze(), cmap='gray')
plt.title(f'Predicted number: {np.argmax(model.predict(X_test[n].reshape(1, 28, 28, 1)))}')
plt.show()
# Evaluate the model
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])