import csv
import cv2
import numpy as np


#keras v1.2.1
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D as Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2

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
                for i in range(3):
                    img_name = row[i].split("/")[-1]
                    current_path = "data/IMG/" + img_name
                    image = cv2.imread(current_path)
                    # Preprocess the image here and append it to the list.
                    image = image[60:140,:] #cv2.resize(image[60:140,:], (64,64))
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
        self.model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(80,320,3)))
        # Add Conv Layers based on the paper https://arxiv.org/pdf/1604.07316v1.pdf
        self.model.add(Conv2D(24,5,5, activation="relu", border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
        # self.model.add(MaxPooling2D())
        self.model.add(Conv2D(36,5,5, activation="relu", border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
        # self.model.add(MaxPooling2D())
        self.model.add(Conv2D(48,5,5, activation="relu", border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
        # self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64,3,3, activation="relu", border_mode='same', subsample =(2,2), W_regularizer = l2(0.001)))
        # self.model.add(MaxPooling2D())
        self.model.add(Conv2D(64,3,3, activation="relu", border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
        # self.model.add(MaxPooling2D())

        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Dense(50))
        self.model.add(Dense(10))
        self.model.add(Dense(1))
        self.model.compile(loss='mse', optimizer='adam')

    def train_model(self):
        self.model.fit(self.x_train, self.y_train, validation_split = 0.2, shuffle = True, nb_epoch = 1)

    def save_model(self):
        self.model.save("model.h5")

data = DataReader("data/driving_log.csv")
x_train = data.get_imgs()
y_train = data.get_measurements()

km = KerasModel(x_train, y_train)
km.make_conv_layers()
km.train_model()
km.save_model()
                                                                                                      