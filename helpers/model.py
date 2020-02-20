__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import random
import numpy as np
import tensorflow.keras as K
import tensorflow.keras.layers as KL

# from _core import normalize_image

def simple_CNN(convolutional_features=32,
               dense_features=128):

    image = KL.Input(shape=[32, 32, 1], name="input")

    conv1 = KL.Conv2D(convolutional_features, 3, activation='relu', padding='valid')(image)
    pool1 = KL.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = KL.Conv2D(convolutional_features, 3, activation='relu', padding='valid')(pool1)
    pool2 = KL.MaxPooling2D(pool_size=(2, 2))(conv2)

    flat1 = KL.Flatten()(pool2)

    logits = KL.Dense(5, activation='linear')(flat1)
    softmax = KL.Activation('softmax', name='output')(logits)

    return K.Model(inputs=image, outputs=softmax, name="ConvolutionalNeuralNetwork")



def data_generator(data, batch_size=32):

    num_images = data['images'].shape[0]

    while True:

        to_use = [n for n in range(num_images)]
        random.shuffle(to_use)

        batch_images, batch_labels = [], []

        while to_use and len(batch_images)<batch_size:

#             print(len(to_use), len(batch_images))

            n = to_use.pop(0)

            image = normalize_image(data['images'][n,...])

            batch_images.append(image[...,np.newaxis])
            batch_labels.append(data['labels'][n])

        yield ({'input': np.stack(batch_images, axis=0)},
               {'output': K.utils.to_categorical(np.stack(batch_labels, axis=0), num_classes=5)})



def train_model(num_epochs=10):

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
    return K.models.load_model('./data/model.h5')


if __name__ == '__main__':
    train_model(num_epochs=50)
