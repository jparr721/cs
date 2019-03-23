import keras
from sklearn.model_selection import train_test_split
import pathlib
import cv2


class CoordiNet():
    def __init__(self, img_x, img_y, input_data_path, label_data_path):
        self.img_x = img_x
        self.img_y = img_y
        self.input_data_path = input_data_path
        self.label_data_path = label_data_path

    def make_net(self, models: list):
        self.model = keras.models.Sequential()
        for model in models:
            self.model.add(model)

    def compile_network(self):
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def initialize_datasets(self):
        input_data = list(self.input_data_path.glob('*.png'))
        label_data = []

        # Read the text file
        with open(self.label_data_path) as lp:
            label_data = lp.readlines()

        # Remove any extra bs
        label_data = [x.strip() for x in label_data]

        # Make our sweet sweet training data
        X_train, X_test, y_train, y_test = train_test_split(
                self.input_data_path,
                self.label_data_path,
                test_size=0.3)

        # Make our sweet sweet validation sets
        X_test, X_valid, y_test, y_valid = train_test_split(
                X_test, y_test, test_size=0.3)

        # My god...
        return X_train, X_test, X_valid, y_train, y_test, y_valid

    def train_network(
            self,
            X_train,
            X_valid,
            y_train,
            y_valid
            epochs=1000,
            batch_size=32,
            save_file='model.hd5'):
        self.model.fit(
            X_train, y_train, validation_data=(
                X-valid, y_valid), epochs=epochs, batch_size=batch_size)

        print(f'Saving model to location: {save_file}')
        self.model.save_weights(save_file)

    def eval(self, X_test, y_test):
        loss, acc = self.model.evaluate(X_test, y_test)
        return loss, acc
