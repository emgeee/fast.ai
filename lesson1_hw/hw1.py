"""
First run (no changes): 360/360 [==============================] - 155s 430ms/step - loss: 0.1149 - acc: 0.9686 - val_loss: 0.0503 - val_acc: 0.9815
Changing Adam optimizer lr to 0.0001: 360/360 [==============================] - 157s 436ms/step - loss: 0.2486 - acc: 0.9133 - val_loss: 0.0666 - val_acc: 0.9760
Remove 1500 images from training set: 180/180 [==============================] - 146s 809ms/step - loss: 0.1191 - acc: 0.9670 - val_loss: 0.0639 - val_acc: 0.9805
using 3000 images in validation set: 149/149 [==============================] - 144s 965ms/step - loss: 0.1385 - acc: 0.9621 - val_loss: 0.0556 - val_acc: 0.9838



"""
import os, json, inspect, sys
# os.environ["THEANO_FLAGS"] = "device=cpu"

sys.path.append('..')

import numpy as np
np.set_printoptions(precision=4, linewidth=100)

import utils
import pprint

from matplotlib import pyplot as plt
from importlib import reload

from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model, load_model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image

PATH = 'data/'
FILES_PATH = 'http://files.fast.ai/models/'
CLASS_FILE = 'imagenet_class_index.json'
VGG_MEAN = np.array([123.68, 116.779, 103.939]).reshape((3, 1, 1))

RETRAINED_FNAME = 'custom_cat_dog.h5'

def pretty_print(data):
    pp = pprint.PrettyPrinter(indent=4)
    print(pp.pprint(data))


def conv_block(layers, model, filters):
    for i in range(layers):
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


def fc_block(model):
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))


def vgg_preprocess(x):
    # subtract main
    x = x - VGG_MEAN

    # reverse axis bgr -> rgb
    return x[:, ::-1]


def vgg_16():
    model = Sequential()
    model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

    conv_block(2, model, 64)
    conv_block(2, model, 128)
    conv_block(3, model, 256)
    conv_block(3, model, 512)
    conv_block(3, model, 512)

    model.add(Flatten())
    fc_block(model)
    fc_block(model)

    model.add(Dense(1000, activation='softmax'))
    return model


def get_batches(dirname,
                gen=image.ImageDataGenerator(),
                shuffle=True,
                batch_size=4,
                class_mode='categorical'):

    return gen.flow_from_directory(PATH+dirname,
                                   target_size=(224, 224),
                                   class_mode=class_mode,
                                   shuffle=shuffle,
                                   batch_size=batch_size)


def configure_model(model, num_labels):
    model.pop()
    # set remaining layers to not be changeable
    for layer in model.layers:
        layer.trainable = False

    model.add(Dense(num_labels, activation='softmax'))
    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def retrain_model(model, train_batches, valid_batches, batch_size=64):
    model.fit_generator(train_batches,
            steps_per_epoch=int(np.ceil(train_batches.samples/batch_size)),
            epochs=1,
            validation_data=valid_batches,
            validation_steps=int(np.ceil(valid_batches.samples/batch_size)))

    model.save(RETRAINED_FNAME)
    return model


def main():
    weight_path = get_file('vgg16.h5',
                           FILES_PATH + 'vgg16.h5',
                           cache_subdir='models')

    batch_size = 128
    train_batches = get_batches('train', batch_size=batch_size)
    valid_batches = get_batches('valid', batch_size=batch_size)

    classes = list(iter(train_batches.class_indices)) # get a list of all the class labels
    print(classes)

    # Assemble a Keras model with the pretrained weights
    model = vgg_16()

    ######################
    # # Load pretrained weights into the model
    # model.load_weights(weight_path)
    # configure_model(model, 2)
    #
    # # Remove the final layer and retrain it using dog/cat data
    # retrain_model(model,
    #               train_batches,
    #               valid_batches,
    #               batch_size)
    #
    ####################3
    model.load_weights(RETRAINED_FNAME)

    print('model and weights loaded...')
    test_batches = get_batches('test', batch_size=batch_size, shuffle=False, class_mode=None)

    filenames = test_batches.filenames
    predictions = model.predict_generator(test_batches)

    # 0 = cat, 1 = dog
    # rounded = np.argmax(predictions, axis=1)
    rounded = predictions

    with open('submission.csv', 'w') as file:
        file.write('id,label\n')
        for idx, prediction in enumerate(rounded):
            id = filenames[idx].replace('foo/', '').replace('.jpg', '')
            str = f'{id},{prediction[1]}'
            print(str)
            file.write(str + '\n')




if __name__ == '__main__':
    main()
