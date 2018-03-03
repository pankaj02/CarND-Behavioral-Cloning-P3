import logging
import csv
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, MaxPooling2D, Cropping2D, Dropout
from keras.layers import Convolution2D


def create_logger():
    logger = logging.getLogger(__name__)

    logger.setLevel(logging.DEBUG)
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    return logger


logger = create_logger()

DRIVING_LOG_CSV = 'driving_log.csv'

images = []
measurements = []


def load_image(center_source_path, measurement, data='data'):
    filename = center_source_path.split('/')[-1]
    current_path = data + '/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurements.append(measurement)


def load_images(data_folder, skip_header=False, left=True, right=True):
    lines = []
    with open('%s/%s' % (data_folder, DRIVING_LOG_CSV)) as csvfile:
        reader = csv.reader(csvfile)
        if skip_header:
            next(reader, None)
        for line in reader:
            lines.append(line)

    for line in lines:
        load_image(line[0], float(line[3]), data_folder)
        if left:
            load_image(line[1], float(line[3]) + 0.2, data_folder)
        if right:
            load_image(line[2], float(line[3]) - 0.2, data_folder)


logger.info('Loading images from data folder')

load_images('data', skip_header=True, left=True, right=True)
logger.info('Total Loaded images %s', len(images))

logger.info('Loading images from to_center folder')

load_images('to_center', skip_header=False, left=True, right=False)
logger.info('Total Loaded images %s', len(images))

logger.info('Loading images from right_to_center folder')

load_images('right_to_center', skip_header=False, left=False, right=False)
logger.info('Total Loaded images %s', len(images))


def augment_images():
    augmented_images, augmented_measurments = [], []
    logger.info('Augementing Images ....')
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurments.append(measurement)
        augmented_images.append(cv2.flip(image, 1))
        augmented_measurments.append(measurement * -1.0)
    return  augmented_images,augmented_measurments


augmented_images, augmented_measurments = augment_images()

logger.info('Total Loaded images after augmentation %s', len(augmented_images))

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurments)


def model():
    """
    Model Description -
    Convolution Layers

    1) 24 5*5 conv layer with Stride of 2 with activation Relu
    2) 36 5*5 conv layer with Stride of 2 with activation Relu
    3) 48 5*5 conv layer with Stride of 2 with activation Relu
    4) 64 3*3 conv layer with activation Relu
    5) 64 3*3 conv layer with activation Relu

    Fully connected Layers
    1) FC Layer - 100 dimension
    2) FC Layer - 50 dimension
    3) Output Layer - 1

    :return:
    """
    model = Sequential()

    #Normalize Data
    model.add(Lambda(lambda x: x / 255.0, input_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))

    model.add(Flatten())

    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(1))

    return model


model = model()


model.compile(loss='mse', optimizer='adam')

logger.info('Running model with error function as mse and optimizer as Adam')

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

logger.info('Saving Model ............')


model.save('model.h5')
