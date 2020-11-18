### YOUR CODE HERE
import tensorflow as tf

from tensorflow.keras import layers, models, Input, Model

"""This script defines the network.
"""

class MyNetwork(object):

    def __init__(self, configs):
        self.configs = configs

    def __call__(self):
        '''
        Args:
            inputs: A Tensor representing a batch of input images.
            training: A boolean. Used by operations that work differently
                in training and testing phases such as batch normalization.
        Return:
            The output Tensor of the network.
        '''
        return self.build_network()

    def build_network(self):

        inp = Input(shape=self.configs["input_size"])

        x = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(64, kernel_initializer="glorot_normal", activation='relu')(x)
        out = layers.Dense(10)(x)

        model = Model(inputs=inp, outputs=out, name="cifar_10_model")

        return model


### END CODE HERE