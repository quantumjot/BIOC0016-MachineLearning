__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import logging
import random
import numpy as np
import tensorflow.keras as K
import tensorflow.keras.layers as KL

# hack to stop too many warnings
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

def simple_CNN(convolutional_kernels=32):
    """ simple_CNN

    Build a simple convolutional classifier, that returns a one-hot
    classification (n,)

    Args:
        convolutional_kernels: the number of kernels for the conv layers

    Returns:
        a keras model of the CNN
    """
    # first, set up the input layer
    image = KL.Input(shape=[None, None, 1], name="input")

    # convolutions and pooling
    conv1 = KL.Conv2D(convolutional_kernels, (3, 3), activation='relu', padding='valid')(image)
    pool1 = KL.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = KL.Conv2D(convolutional_kernels, (3, 3), activation='relu', padding='valid')(pool1)
    pool2 = KL.MaxPooling2D(pool_size=(2, 2))(conv2)

    # flatten the data and fully connect to output layer
    flat1 = KL.Flatten()(pool2)
    logits = KL.Dense(5, activation='linear')(flat1)
    softmax = KL.Activation('softmax', name='output')(logits)

    # return the full model
    return K.Model(inputs=image, outputs=softmax, name="Convolutional_Neural_Network")




def data_generator(data, batch_size=32):
    """ generator function to provide data to the network while training

    TODO(arl): this does no augmentation. should really do that for future
        iterations of this practical
    """
    num_images = data['images'].shape[0]

    # repeat forever
    while True:
        to_use = [n for n in range(num_images)]
        random.shuffle(to_use)

        batch_images, batch_labels = [], []

        # get a batch
        while to_use and len(batch_images)<batch_size:
            n = to_use.pop(0)
            image = normalize_image(data['images'][n,...])
            batch_images.append(image[...,np.newaxis])
            batch_labels.append(data['labels'][n])

        yield ({'input': np.stack(batch_images, axis=0)},
               {'output': K.utils.to_categorical(np.stack(batch_labels, axis=0), num_classes=5)})



def train_model(num_epochs=10):
    """ Train the CNN model """
    data = np.load('../data/training_data.npz')

    batch_size = 32
    num_training_images = data['images'].shape[0]

    model = simple_CNN()
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    model.fit(data_generator(data, batch_size=batch_size),
              epochs=num_epochs, steps_per_epoch=num_training_images / batch_size)

    model.save('../data/model.h5')



def load_model():
    """ load the pre-trained model """
    return K.models.load_model('./data/model.h5')



def visualize_layers(m, x, layer=1):
    """ return the layer activations from the model """
    assert(layer in [1,3])
    partial_model = K.Model(inputs=m.inputs, outputs=m.layers[layer].output)
    return partial_model.predict(x, batch_size=1)


if __name__ == '__main__':
    train_model(num_epochs=50)
