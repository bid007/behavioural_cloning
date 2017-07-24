import csv
import cv2
import numpy as np


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


class DataReader:
    """This class reads the file and returns required data"""
    def __init__(self, filename):
        self.filename = filename
        self.read_file()


    def read_file(self):
        imgs = []
        angles = []

        with open(self.filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                img_name = row[0].split("/")[-1]
                current_path = "data/IMG/" + img_name
                image = cv2.imread(current_path)
                imgs.append(image)
                imgs.append(cv2.flip(image, 1))
                steer_angle = float(row[3])
                angles.append(steer_angle)
                angles.append(steer_angle * -1)

        self.imgs = np.array(imgs)
        self.measurements = np.array(angles)

    def get_imgs(self):
        return self.imgs

    def get_measurements(self):
        return self.measurements

class KerasModel(object):
    def __init__(self, x_train, y_train):
        self.model = None
        self.x_train = x_train
        self.y_train = y_train

    def make_conv_layers(self):
        self.model = Sequential()
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
        # Add Conv Layers based on the paper https://arxiv.org/pdf/1604.07316v1.pdf
        self.model.add(Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=36, kernel_size=(5,5), strides=(2,2), activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=48, kernel_size=(5,5), strides=(2,2), activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
        self.model.add(MaxPooling2D())
        self.model.add(Conv2D(filters=64, kernel_size=(3,3), activation="relu"))
        self.model.add(MaxPooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)

    def save_model(self):
        self.model.save("model.h5")

data = DataReader("data/driving_log.csv")
x_train = data.get_imgs()
y_train = data.get_measurements()

km = KerasModel(x_train, y_train)
km.make_conv_layers()
km.train_model()
km.save_model()
                                                                                                      