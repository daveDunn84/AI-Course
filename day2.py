#import necessary libraries
import numpy as np #for numerical operations in python
import tensorflow as tf #for deep learning in python
from tensorflow.keras import datasets, layers, models #datasets includes common datasets (e.g. MNIST)
from tensorflow.keras.utils import to_categorical #to convert labels to categorical format
import matplotlib.pyplot as plt

#load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data() #returns a tuple of training and testing data

# pre-processing: normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape the images to (28, 28, 1) <- H, W, Colour channels
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))

# convert the labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# build the cnn model <- CNN = convolutional neural network
model = models.Sequential()

# first convolutional layer
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

# second convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(2, 2))

# third convolutional layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# flatten the 3D output to 1D and add a Dense layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# output layer with 10 neurons (fpr 10 digit classes)
model.add(layers.Dense(10, activation='softmax'))

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train the model
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# make predictions on test images
predictions = model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[0])}")

plt.imshow(test_images[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted label: {predictions[0].argmax()}")
plt.show()