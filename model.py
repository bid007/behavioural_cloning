import csv
import cv2
import numpy as np

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


data = DataReader("data/driving_log.csv")
x_train = data.get_imgs()
y_train = data.get_measurements()

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6,5,5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 5)
model.save('model.h5')
                                                                                                      