from keras.models import Sequential
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
from keras.datasets import cifar10  # useful for now, should be changed later

(X, y), (X_test, y_test) = cifar10.load_data()

# Normalize the RGB data
X, X_test = X.astype('float32')/255.0, X_test.astype('float32')/255.0

# One-hot encode the y vectors
# cifar10 data has 10 classes
y, y_test = u.to_categorical(y, 10), y.to_categorical(y_test, 10)

model = Sequential()
# This is our 32x32 supporting conv net, for different data we'll need to
# change this. Right now it support 3 channels.
# padding=same means the input and output dimensions are the same
model.add(
        Conv2D(
            32, (3, 3), (1, 1),
            input_shape=(32, 32, 3),
            padding='same',
            activation='relu'))

# Use dropout to prevent accidentally overfitting on the network
model.add(Dropout(0.2))

# Add another model with padding of valid which allows the output
# to take any shape. This is because we are going to pool the layers
# before feeding them into the full connected unit and, as a result,
# we may want to change shape
model.add(Conv2D(32, (3, 2), activation='relu', padding='valid'))

# Add our max pooling layer with a size of 2x2 to feed to our
# fully connected layer
model.add(MaxPooling2D((2, 2)))

# We must flatten our data before the network can read it. Multi-dim
model.add(Flatten())

# tensors cannot be read by dense layers
model.add(Dense(512, activation='relu'))  # units = 32x32 / 2

# Drop 30% of nodes
model.add(Dropout(0.3))

# Add a dense layer with our final 10 class labels
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(momentum=0.5, decay=0.0004, metrics=['accuracy']))

# Train the system
model.fit(X, y, validation_data=(X_test, y_test), epochs=25,
          batch_size=512)
# Save the weights to use for later
model.save_weights("trained.hdf5")
# Finally print the accuracy of our model!
print('Accuracy: {}'.format(model.evaluate(X_test, y_test)[1]*100))
