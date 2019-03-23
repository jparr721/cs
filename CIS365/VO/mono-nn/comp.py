#!/usr/bin/env python3

import keras
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import collections
import bisect
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

img_x, img_y = 32, 32


# Take an image filename & return the normalized numpy array
def image_to_np(filename):
    image = cv2.imread(filename)
    image = cv2.resize(
            image, dsize=(img_x, img_y), interpolation=cv2.INTER_CUBIC)
    return image/255


# Plot predicted vs actual absolute positions
def show_plots(model, dimages, dpositions):
    dpredictions = model.predict(dimages)
    dactual = dpositions

    fig = plt.figure()
    fig2 = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ex = fig2.add_subplot(111, projection='3d')

    predictions = []
    actual = []
    x, y, z = [], [], []
    x2, y2, z2 = [], [], []
    for dpred, dact in zip(dpredictions, dactual):
        predictions.append([x, y, z])
        actual.append([x2, y2, z2])
        x += [dpred[0]]
        y += [dpred[1]]
        z += [dpred[2]]
        x2 += [dact[0]]
        y2 += [dact[1]]
        z2 += [dact[2]]

    ax.plot(x, y, z, color="red")
    ex.plot(x2, y2, z2)

    plt.show()


# Process data from a tgz to d_images & d_positions
def process_source(root):
    data_root = pathlib.Path('./{}'.format(root))
    image_root = data_root / 'rgb'
    image_paths = list(image_root.glob('*.png'))
    image_paths = sorted([str(path) for path in image_paths])
    images = [image_to_np(f) for f in image_paths]

    # Get the labels (filenames) and data
    f = open(data_root / 'groundtruth.txt', 'r')
    motion_data = f.readlines()[3:]
    f = open(data_root / 'rgb.txt', 'r')
    label_data = f.readlines()[3:]

    pat = collections.OrderedDict()
    for datum in motion_data:
        pat[float(datum.split()[0])] = \
            np.array([float(n) for n in datum.split()[1:4]])

    def position_at_time(timestamp):
        ind = bisect.bisect_left(list(pat.keys()), timestamp)
        return list(pat.items())[ind]

    positions = []
    for label in label_data:
        timestamp = float(label.split()[0])
        real_time, closest_position = position_at_time(timestamp)
        positions.append(closest_position)

    # Convert images & positions to their delta values
    # (differences between adjacent frames & positions)
    d_images = []
    d_positions = []
    for i in range(len(images)-1):
        d_image = images[i+1] - images[i]
        d_images.append(d_image)
        d_position = positions[i+1] - positions[i]
        d_positions.append(d_position)

    return d_images, d_positions


if __name__ == '__main__':
    # Define the model
    input_shape = (img_x, img_y, 3)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(
            img_x,
            (3, 3),
            padding='same',
            activation='relu',
            input_shape=(img_x, img_y, 3)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(
            img_x,
            (3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(
            img_x*2,
            (3, 3),
            padding='same',
            activation='relu'
        ),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0, 2),
        tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_constraint=keras.constraints.maxnorm(3)
        ),
        tf.keras.layers.Dense(
            512,
            activation='relu',
            kernel_constraint=keras.constraints.maxnorm(3)
        ),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3)
    ])

    model.compile(optimizer='adam',
                  loss='mean_squared_error',
                  metrics=['accuracy'])

    source = 'rgbd_dataset_freiburg1_xyz',

    all_d_images = []
    all_d_positions = []
    d_i, d_p = process_source(source)
    all_d_images += d_i
    all_d_positions += d_p
    all_d_images = np.array(all_d_images)
    all_d_positions = np.array(all_d_positions)

    # Split into training, testing, & validation sets
    x_train, x_test, y_train, y_test = train_test_split(
            all_d_images, all_d_positions, test_size=0.3)
    x_test, x_valid, y_test, y_valid = train_test_split(
            x_test, y_test, test_size=0.2)

    # Train the model
    model.fit(
            x_train, y_train, validation_data=(
                x_valid, y_valid), epochs=500, batch_size=32)

    # model.load_weights("model.h5")

    show_plots(model, all_d_images, all_d_positions)

    # Print accuracy on test set
    loss, acc = model.evaluate(x_test, y_test)

    model.save_weights("model.h5")

    print(loss, acc)
