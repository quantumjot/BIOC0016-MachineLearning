__author__ = "Alan R. Lowe"
__email__ = "a.lowe@ucl.ac.uk"

import os
import random
import logging

import numpy as np
from skimage import io
from skimage.filters import median as median_filter
from skimage.transform import resize

import matplotlib.pyplot as plt

from .model import load_model

CHANNEL_ORDER = ['Brightfield', 'GFP', 'RFP']
STATES = ['interphase', 'prometaphase', 'metaphase', 'anaphase', 'apoptosis', 'unknown']
MAX_IMAGE_REQUEST = 100





class _DatasetContainer:
    def __init__(self):
        self.__data = np.load('./data/test_data.npz')['images']
        self.__idx = [i for i in range(self.__data.shape[0])]

    def __len__(self):
        return self.__data.shape[0]

    def get_random(self, num_images=1):
        assert(num_images>0 and num_images<MAX_IMAGE_REQUEST)
        random.shuffle(self.__idx)
        images = [normalize_image(self.__data[i,...]) for i in self.__idx[:num_images]]
        return images, self.__idx[:num_images]



# instantiate the dataset container
dataset = _DatasetContainer()

def get_example_images(num_images=1):
    return dataset.get_random(num_images=num_images)



def normalize_image(im):
    assert(isinstance(im, np.ndarray))
    assert(im.ndim == 2)
    im = im.astype(np.float32)
    mu, std = np.mean(im), np.std(im)
    normed = (im - mu) / np.max([std, 1./np.prod(im.shape)])
    return normed



def remove_outliers(im):
    assert(isinstance(im, np.ndarray))
    if im.ndim == 3:
        for channel in range(im.shape[-1]):
            med_filter = median_filter(im[...,channel])
            mask = im[...,channel] > med_filter + 2.*np.std(med_filter)
            im[...,channel][mask] = med_filter[mask]
    return im



def encode_images(path, num_images=1000):
    """ encode images as a numpy file, with accompanying json file """

    files = []

    for state in STATES:
        fp = os.path.join(path, state)
        files += [(state, fp, f) for f in os.listdir(fp) if f.endswith('.tif')]

    # shuffle the images
    random.shuffle(files)

    images, labels = [], []

    for state, fp, file in files:

        # the the filename and label
        fn, ext = os.path.splitext(file)
        label = STATES.index(state)

        # get the cell type, and extract only that fluorescence channel
        cell_channel = CHANNEL_ORDER.index(fn[-3:])

        im = io.imread(os.path.join(fp, file))
        im = remove_outliers(im)

        # rescale the image
        im_resized = resize(im, (32,32), preserve_range=True)
        images.append(im_resized[...,cell_channel].astype('uint8'))
        labels.append(label)

    # make this into a large numpy array for saving
    np_images_annotation = np.stack(images[:num_images], axis=0)
    np_images_training = np.stack(images[num_images:], axis=0)
    np_images_training_labels = np.stack(labels[num_images:], axis=0)

    print(np_images_annotation.shape,
          np_images_training.shape,
          np_images_training_labels.shape)

    # write out the numpy array
    np.savez('./data/cell_data.npz', images=np_images_annotation)
    np.savez('./data/training_data.npz',
             images=np_images_training,
             labels=np_images_training_labels)

    # write out the mapping to the original files
    with open('./data/cell_data.txt', 'w') as file:
        for label, fp, fn in files[:num_images]:
            file.write(f'{fp} {fn} \n')

    return images



def plot_images(images, image_idx, cols=5):
    """ plot the images and their ID """
    num_images = len(images)
    rows = np.ceil(num_images/cols)

    fig_height = 3*rows

    plt.figure(figsize=(15, fig_height))
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(f'Image #{image_idx[i]}')
        plt.axis('off')
    plt.show()



def validate_annotation(annotation):
    """ validate the student annotation """

    validate = True

    # check that this is a dictionary
    if not isinstance(annotation, dict):
        logging.error('The annotation must be a dictionary')
        validate = False

    # check that it has the correct keys
    if not all([k in STATES for k in annotation.keys()]):
        unknown = ', '.join([str(k) for k in annotation.keys() if k not in STATES])
        logging.error(f'Dictionary keys not recognized: {unknown}')
        validate = False

    # check to make sure the numbers make sense
    values = list(annotation.values())
    merged = []
    while values:
        merged += values.pop(0)

    # check that all entries are integers
    if not all([isinstance(v, int) for v in merged]):
        logging.error('Annotation lists must be integer numbers')
        validate = False
    else:
        # check that they fall within a useful range
        if not all([v>=0 and v<len(dataset) for v in merged]):
            logging.error('Annotation outside of range')
            validate = False

    # finally, check for duplications
    unique = list(set(merged))
    if any([merged.count(v)>1 for v in unique]):
        logging.error('Duplicate annotation found')
        validate = False

    if not validate:
        raise ValueError('Validation failed')

    return validate




def load_CNN_model():
    return load_model()
