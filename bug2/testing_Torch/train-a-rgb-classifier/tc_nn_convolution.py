#!/usr/bin/env python2

"""Testing a NN with flash cards."""

# pylint: disable=unused-import

import logging
import itertools
import numpy as np
from scipy.misc import imresize
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam
from keras.models import Sequential
from keras.layers.convolutional import (Convolution2D, ZeroPadding2D,
                                        MaxPooling2D)
from keras import backend as K


class NNHyperParameters(object):
    """Container class for some NN hyper-parameters."""

    available_keras_optimizers = []
    available_keras_objectives = []
    available_keras_activations = []
    available_keras_initializations = []
    available_keras_border_modes = []

    @classmethod
    def get_available_optimizers(cls):
        """Get all the available optimizers available from Theano or
        TensorFlow through Keras.

        http://keras.io/optimizers/
        """

        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        cls.available_keras_optimizers.append(sgd)

        rms_prop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06)
        cls.available_keras_optimizers.append(rms_prop)

        ada_grad = Adagrad(lr=0.01, epsilon=1e-06)
        cls.available_keras_optimizers.append(ada_grad)

        ada_delta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
        cls.available_keras_optimizers.append(ada_delta)

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        cls.available_keras_optimizers.append(adam)

    @classmethod
    def get_available_objectives(cls):
        """Get all the available objectives available from Theano or
        TensorFlow through Keras.

        http://keras.io/objectives/
        """

        cls.available_keras_objectives = ['mean_squared_error',
                                          'root_mean_squared_error',
                                          'mean_absolute_error',
                                          'mean_absolute_percentage_error',
                                          'mean_squared_logarithmic_error',
                                          'squared_hinge',
                                          'hinge',
                                          'binary_crossentropy',
                                          'poisson_loss']

    @classmethod
    def get_available_activations(cls):
        """Get all the available activations available from Theano or
        TensorFlow through Keras.

        http://keras.io/activations/
        """

        cls.available_keras_activations = ['softplus',
                                           'relu',
                                           'tanh',
                                           'sigmoid',
                                           'hard_sigmoid',
                                           'linear']

    @classmethod
    def get_available_initializations(cls):
        """Get all the available initializations available from Theano or
        TensorFlow through Keras.

        http://keras.io/initializations/
        """

        cls.available_keras_initializations = ['uniform',
                                               'lecun_uniform',
                                               'normal',
                                               'identity',
                                               'orthogonal',
                                               'zero',
                                               'glorot_normal',
                                               'glorot_uniform',
                                               'he_normal',
                                               'he_uniform']

    @classmethod
    def get_available_border_modes(cls):
        """Get all the available border-modes available from Theano or
        TensorFlow through Keras. It is for Convolutional layers:

        http://keras.io/layers/convolutional/
        """

        cls.available_keras_border_modes = ['valid', 'same']


def main():
    """Main program."""

    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    logging.debug("Preparing some combinations of NN hyper-parameters")

    NNHyperParameters.get_available_optimizers()
    NNHyperParameters.get_available_objectives()
    NNHyperParameters.get_available_activations()
    NNHyperParameters.get_available_initializations()
    NNHyperParameters.get_available_border_modes()

    all_combinations_hyp_param = list(itertools.product(
        NNHyperParameters.available_keras_optimizers,
        NNHyperParameters.available_keras_objectives,
        NNHyperParameters.available_keras_activations,
        NNHyperParameters.available_keras_initializations,
        NNHyperParameters.available_keras_border_modes))

    logging.debug("Testing hyper-parameters")

    for combination in all_combinations_hyp_param:
        optimizer = combination[0]
        objective = combination[1]
        activation = combination[2]
        initialization = combination[3]
        border_mode = combination[4]

        # we need to protect the call to the NN with these hyperparameters in
        # this iteration in the loop because we don't know if this combination
        # of hyper-parameters is supported
        try:
            nn_model = create_nn_model_convolut(optimizer, objective,
                                                activation, initialization,
                                                border_mode)
            # TODO: train the NN
        except Exception as exc:        # pylint: disable=broad-except
            logging.error("Exception caught %s for hyper-parameters: %s",
                          str(exc), ' '.join(repr(combination)))


class FlashCards(object):
    """Container class for some flash-cards parameters."""
    # pylint: disable=too-few-public-methods

    # the flash cards don't need to squares, because we use vertical bars, not
    # horizontal ones
    flash_card_dims = (100, 20)

    # to make the flash-cards distinct to reinforce learning
    variation_in_width = 10


def create_nn_model_convolut(optimizer, objective, activation, initialization,
                             border_mode):
    """Creates a Conv with the hyper-parameters given, taking as an
    inspiration the Keras example of 'deep_dream.py'.

    Returns the NN schema built."""

    logging.debug("Building a NN schema with %s %s %s %s %s", repr(optimizer),
                  repr(objective), repr(activation), repr(initialization),
                  repr(border_mode))

    (img_width, img_height) = FlashCards.flash_card_dims

    input_img = K.placeholder((1, 3, img_width, img_height))

    # build the VGG16 network
    first_layer = ZeroPadding2D((1, 1),
                                input_shape=(3, img_width, img_height))
    first_layer.input = input_img

    model = Sequential()
    model.add(first_layer)
    # TODO: not all layers have to have the same hyperparameters (activation,
    # etc), so a generic constructor is not fit for all of them, ie., each
    # layer can have different hyperparameters
    model.add(Convolution2D(64, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode=border_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode=border_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode=border_mode))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation=activation,
                            init=initialization, border_mode=border_mode))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), border_mode=border_mode))

    model.compile(loss=objective, optimizer=optimizer)

    logging.debug("Built NN schema with %s %s %s %s %s", repr(optimizer),
                  repr(objective), repr(activation), repr(initialization),
                  repr(border_mode))

    return model    # return the NN schema


def generate_flash_cards(colors_seq, dest_flash_card_preffix):
    """Generates a set of flash cards .PNG files, all having this same color
       sequence 'colors_seq' from left to right, but differing among them, not
       in the colors, but only in the width of each color bar in the flash
       card. This is necessary to increase the dimensionality of the tensor
       for reinforcement learning. All flash cards are saved with a common
       preffix 'dest_flash_card_preffix'.
    """

    from random import shuffle

    uniform_width_of_each_bar = int(FlashCards.flash_card_dims[0] /
                                    len(colors_seq))

    # the shifts in the column widths of each color bar inside the flash card:
    # it has to be [-a, 0] instead of [a/2, a/2+1], otherwise there is a
    # risk the last bar in the flash card is left without space
    shifts_to_reinforce_learning = list(
        range(-FlashCards.variation_in_width, 0)
    )

    shuffle(shifts_to_reinforce_learning)

    for variation in xrange(len(shifts_to_reinforce_learning)):
        dest_flash_card_fname = "{}_{}.png".format(dest_flash_card_preffix,
                                                   variation)
        column_width = (uniform_width_of_each_bar +
                        shifts_to_reinforce_learning[variation])

        generate_a_flash_card(colors_seq, column_width, dest_flash_card_fname)


def generate_a_flash_card(colors_seq, color_bar_width,
                          dummy_dest_flash_card_fname):
    """Generate one flash card with the given colors seq, width of each color
    bar, to the given 'dest_flash_card_fname' PNG file."""

    from PIL import Image, ImageDraw

    a_flash_card = Image.new('RGB', FlashCards.flash_card_dims)
    draw = ImageDraw.Draw(a_flash_card)

    current_column = 0

    for color in colors_seq:
        # paint this bar on the flash card with the current color in
        # colors_seq
        next_col = current_column + color_bar_width
        draw.rectangle([(current_column, 0),
                        (next_col, FlashCards.flash_card_dims[1])],
                       outline=color, fill=color)
        current_column = next_col

    # paint the remainder bar in the flash card with the last color, if
    # any remainder bar was left because of the shift on the uniform width
    if current_column < FlashCards.flash_card_dims[0] - 1:
        draw.rectangle([(current_column, 0), FlashCards.flash_card_dims],
                       outline=colors_seq[-1], fill=colors_seq[-1])

    # don't save the flash card to file, but leave it on memory: this assumes
    # that the returned PIL image will be passed directly to
    # preprocess_image() below
    # a_flash_card.save(dest_flash_card_fname, 'PNG')
    return a_flash_card


# Similar in idea as in keras/example/deep_dream.py, by Francois Chollet, but
# working directly on PIL images instead of an image_path
def preprocess_image(pil_image, to_img_width, to_img_height):
    """Util function to resize and format pictures into appropriate
    tensors."""
    img = imresize(np.asarray(pil_image), (to_img_width, to_img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img = np.expand_dims(img, axis=0)
    return img


if __name__ == '__main__':
    main()
