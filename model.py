import csv
import cv2
import numpy as np
import random
import pandas as pd
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import math
import matplotlib.image as mpimg
import time
import sys
import argparse

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
        self.colnames = ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']
        self.read_file()
        
    def read_file(self):
        self.data = pd.read_csv('data/driving_log.csv', skiprows=[0], names=self.colnames)
        self.center = self.data.center.tolist()
        center_recover = self.data.center.tolist()
        self.left = self.data.left.tolist()
        self.right = self.data.right.tolist()
        self.steering = self.data.steering.tolist()
        steering_recover = self.data.steering.tolist()
        self.center, self.steering = shuffle(self.center, self.steering)
        self.center, self.X_valid, self.steering, self.y_valid = train_test_split(self.center, self.steering,\
         test_size = 0.20, random_state = 100) 
        self.d_straight, self.d_left, self.d_right = [], [], []
        self.a_straight, self.a_left, self.a_right = [], [], []
        for i in self.steering:
            index = self.steering.index(i)
            if i > 0.15:
                self.d_right.append(self.center[index])
                self.a_right.append(i)
            if i < -0.15:
                self.d_left.append(self.center[index])
                self.a_left.append(i)
            else:
                self.d_straight.append(self.center[index])
                self.a_straight.append(i)

        ds_size, dl_size, dr_size = len(self.d_straight), len(self.d_left), len(self.d_right)
        main_size = int(math.ceil(len(center_recover)))
        l_xtra = ds_size - dl_size
        r_xtra = ds_size - dr_size
        # Generate random list of indices for left and right recovery images
        indice_L = random.sample(range(main_size), l_xtra)
        indice_R = random.sample(range(main_size), r_xtra)

        # Filter angle less than -0.15 and add right camera images into driving left list, minus an adjustment angle #
        for i in indice_L:
            if steering_recover[i] < -0.15:
                self.d_left.append(self.right[i])
                self.a_left.append(steering_recover[i] - 0.27)

        # Filter angle more than 0.15 and add left camera images into driving right list, add an adjustment angle #  
        for i in indice_R:
            if steering_recover[i] > 0.15:
                self.d_right.append(self.left[i])
                self.a_right.append(steering_recover[i] + 0.27)
        X_train = self.d_straight + self.d_left + self.d_right
        y_train = np.float32(self.a_straight + self.a_left + self.a_right)
        # with open(self.filename) as f:
        #     reader = csv.reader(f)
        #     next(reader)
        #     for row in reader:
        #         for i in range(3):
        #             img_name = row[i].split("/")[-1]
        #             current_path = "data/IMG/" + img_name
        #             image = cv2.imread(current_path)
        #             # Preprocess the image here and append it to the list.
        #             image = self.crop_image(self.random_brightness(image))
        #             imgs.append(image)
        #             imgs.append(cv2.flip(image, 1))
        #             steer_angle = float(row[3])
        #             angles.append(steer_angle)
        #             angles.append(steer_angle * -1)
        imgs,angles = [],[]
        for i,each in enumerate(X_train):
            cv_img = mpimg.imread("data/" + each.strip())
            cv_img = self.crop_image(self.random_brightness(cv_img))
            angle = y_train[i]*(1+ np.random.uniform(-0.10,0.10))
            imgs.append(cv_img)
            angles.append(angle)
            imgs.append(cv2.flip(cv_img,1))
            angles.append(angle * -1)

        self.imgs = np.array(imgs)
        self.measurements = np.array(angles)

    def get_imgs(self):
        return self.imgs

    def get_measurements(self):
        return self.measurements

    def random_brightness(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        rand = random.uniform(0.3,1.0)
        hsv[:,:,2] = rand*hsv[:,:,2]
        new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return new_img 

    def crop_image(self, image):
        return image[60:140,:]

class KerasModel(object):
    def __init__(self, x_train, y_train, epoch):
        self.model = None
        self.x_train = x_train
        self.y_train = y_train
        self.epoch = epoch

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
        self.model.fit(self.x_train, self.y_train, validation_split = 0.2, shuffle = True, nb_epoch = self.epoch)

    def save_model(self):
        self.model.save("model.h5")

    def get_model_summary(self):
        return self.model.summary()

parser = argparse.ArgumentParser(description='Run the End-to-End Driving Model.')
parser.add_argument("--epoch", help="The # of iterations to run.", type=int)
parser.add_argument("--benchmark", help="Benchmark the Model??", action="store_true")
args = parser.parse_args()

if not args.epoch:
    epoch = 1
else:
    epoch = args.epoch

if not args.benchmark:
    benchmark = False
else:
    benchmark = True

data = DataReader("data/driving_log.csv")
x_train = data.get_imgs()
y_train = data.get_measurements()

km = KerasModel(x_train, y_train, epoch)
km.make_conv_layers()

if benchmark:
    start_time = time.time()

km.train_model()

if benchmark:
    end_time = time.time()
    print("Total Training time ", end_time-start_time," seconds")

km.save_model()                                                                                                  