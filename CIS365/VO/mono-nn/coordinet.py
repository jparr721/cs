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
        self.model.add(models)

    def train_network(self, epochs: int):
        self.model.compile(optimizer='adam',
                           loss='mean_squared_error',
                           metrics=['accuracy'])

    def initialize_datasets(self):
        input_data = list(self.input_data_path.glob('*.png'))
        X_train, X_test, y_train, y_test = train_test_split(
                self.input_data_path,
                self.label_data_path,
                test_size=0.3)
        X_test_, X_valid, y_test, y_valid = train_test_split(
