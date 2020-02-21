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

from .model import load_model as _load_model
from .model import visualize_layers as _visualize_layers

CHANNEL_ORDER = ['Brightfield', 'GFP', 'RFP']
STATES = ['interphase', 'prometaphase', 'metaphase', 'anaphase', 'apoptosis', 'unknown']
PRED_STATES = STATES[0:5]
MAX_IMAGE_REQUEST = 100

class Image:
    """ Image

    A wrapper for the numpy array data, such that we can store metadata
    including the ID, and allow for small modifications of the data.
    """
    def __init__(self, data, ID):
        assert(isinstance(data, np.ndarray))
        assert(data.ndim == 2)
        assert(data.dtype == np.float32)

        self.__data = data
        self.__ID = ID

    @property
    def ID(self): return self.__ID

    @property
    def shape(self): return self.__data.shape

    @property
    def data(self): return self.__data

    def as_tensor(self):
        return self.__data[np.newaxis, ..., np.newaxis]

    def plot(self):
        plt.imshow(self.data)



class _DatasetContainer:
    def __init__(self):
        self.__data = np.load('./data/test_data.npz')['images']
        self.__idx = [i for i in range(self.__data.shape[0])]

    def __len__(self):
        return self.__data.shape[0]

    def get_random(self, num_images=1):
        assert(num_images>0 and num_images<MAX_IMAGE_REQUEST)
        random.shuffle(self.__idx)
        return [Image(normalize_image(self.__data[i,...]), self.__idx[i]) for i in self.__idx[:num_images]]



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



def plot_images(images, cols=5):
    """ plot the images and their ID """
    num_images = len(images)
    rows = np.ceil(num_images/cols)

    fig_height = 3*rows

    plt.figure(figsize=(15, fig_height))
    for i in range(num_images):
        plt.subplot(rows, cols, i+1)
        plt.imshow(images[i].data, cmap=plt.cm.gray)
        plt.title(f'Image #{images[i].ID}')
        plt.axis('off')
    plt.show()


def get_label_from_ID(annotation, image):
    for k in annotation:
        if image.ID in annotation[k]:
            return k
    return None


def plot_predictions(images, predictions, annotation=None):
    num_images = min(len(images), 5)
    for i in range(num_images):
        plt.figure(figsize=(15,5))
        plt.subplot(1,2,1)
        plt.imshow(images[i].data, cmap=plt.cm.gray)
        plt.title(f'Image #{images[i].ID}')
        plt.axis('off')
        plt.subplot(1,2,2)
        rect = plt.barh(np.arange(len(PRED_STATES)),
                        predictions[i,:],
                        align='center',
                        height=0.75,
                        tick_label=[s.capitalize() for s in PRED_STATES])

        pred = STATES[int(np.argmax(predictions[i,:]))].capitalize()
        ground_truth = None
        if annotation is not None:
            ground_truth = get_label_from_ID(annotation, images[i])
            if ground_truth:
                y = STATES.index(ground_truth)
                if y<5: rect[y].set_facecolor('red')    # can't plot unknown

        plt.xlim([0., 1.1])
        plt.xlabel('P(label|data)')
        plt.title(f'Prediction: {pred}, Ground Truth: {ground_truth}')
        plt.show()


def visualize_layers(m, image, layer=0):
    """ visualize a layer of the network """
    assert(layer in [0,1])
    assert(isinstance(image, Image))
    imtensor = image.as_tensor()

    # now do the prediction
    real_layer = 2*layer+1
    pred = _visualize_layers(m, imtensor, real_layer)

    # make a montage of the activations
    montage = np.zeros((4*pred.shape[1], 8*pred.shape[2]), dtype=np.float32)
    for activation in range(pred.shape[-1]):
        x = (activation % 8)*pred.shape[1]
        y = int(activation/8)*pred.shape[2]
        xs = slice(x, x+pred.shape[1], 1)
        ys = slice(y, y+pred.shape[1], 1)
        montage[ys, xs] = np.squeeze(pred[...,activation])

    fig = plt.figure(figsize=(18,8))
    ax = plt.subplot2grid((1, 3), (0, 0))
    ax.imshow(image.data, cmap=plt.cm.gray)
    ax.set_title(f'Image #{image.ID}')
    ax.set_axis_off()

    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)
    im = ax2.imshow(montage)
    ax2.set_axis_off()
    fig.colorbar(im, orientation="horizontal", pad=0.01)
    ax2.set_title(f'Convolutional layer {layer} activations')
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
    return _load_model()
