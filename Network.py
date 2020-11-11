### YOUR CODE HERE
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.optimizers import *

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
        inp = Input(self.configs["input_size"])
        # Encoder
        x = Convolution2D(64, 3, padding = "same")(inp)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Convolution2D(128, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Convolution2D(256, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        x = Convolution2D(512, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        # Decoder
        x = Convolution2D(512, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(256, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(128, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = UpSampling2D(size=(2, 2))(x)
        x = Convolution2D(64, 3, padding = "same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        
        x = Dense(self.configs["num_classes"])
        x = Activation("softmax")(x)
        model = Model(inputs = inp, outputs = x)


### END CODE HERE