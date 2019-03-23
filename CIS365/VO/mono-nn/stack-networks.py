#!/usr/bin/env python3

import keras
from keras.layers import Dropout, Dense, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils as u
import cv2
from coordinet import CoordiNet
from PIL import Image


if __name__ == '__main__':
    image_path = '../grey/dataset/sequences/00/image_0'
    label_path = '../poses/dataset/poses/00.txt'

    im_dim = im.open(image_path + '/000000.png')
    width, height = im_dim.size

    input_layer_size = width * height

    cn = CoordiNet(image_path, label_path)
    cv1 = Conv2D(
            input_layer_size, (3, 3), (1, 1),
            input_shape=(input_layer_size, input_layer_size, 3),
            padding='same',
            activation='relu')
    dropout1 = Dropout(0.2)
    cv2 = Conv2D(
            input_layer_size, (3, 3),
            activation='relu',
            paddind='valid')
    mp1 = MaxPooling((2, 2))
    flat1 = Flatten()
    dense1 = Dense(input_layer/2, activation='relu')
    dropout2 = Dropout(0.3)
    dense2 = Dense(10, activation='softmax')

    layers = [
            cv1,
            dropout1,
            cv2,
            mp1,
            flat1,
            dense1,
            dropout2,
            dense2
            ]
    cn.make_net(layers)
    cn.compile_network()

    # Initialize our starting datasets
    X_train, X_test, X_valid, y_train, y_test, y_valid = \
            cn.initialize_datasets()

    cn.train_network(X_train, X_valid, y_train, y_valid)
    loss, acc = cn.eval(X_test, y_test)

    print(f'loss: {loss}, accuracy: {acc}')
