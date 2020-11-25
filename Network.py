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

        # Version 1
        if self.configs["version"] == "v1":
            conv_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal", padding="same")(inp)
            dropout_1 = layers.Dropout(rate=self.configs["dropout"])(conv_1)
            batch_norm_1 = layers.BatchNormalization()(dropout_1)
            max_pool_1 = layers.MaxPooling2D((2, 2))(batch_norm_1)
            conv_2 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_1)
            dropout_2 = layers.Dropout(rate=self.configs["dropout"])(conv_2)
            batch_norm_2 = layers.BatchNormalization()(dropout_2)
            max_pool_2 = layers.MaxPooling2D((2, 2))(batch_norm_2)
            conv_3 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_2)
            dropout_3 = layers.Dropout(rate=self.configs["dropout"])(conv_3)
            batch_norm_3 = layers.BatchNormalization()(dropout_3)
            conv_out = layers.MaxPooling2D((2, 2))(batch_norm_3)

        # Version 2
        if self.configs["version"] == "v2":
            conv_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal", padding="same")(inp)
            dropout_1 = layers.Dropout(rate=self.configs["dropout"])(conv_1)
            batch_norm_1 = layers.BatchNormalization()(dropout_1)
            max_pool_1 = layers.MaxPooling2D((2, 2))(batch_norm_1)
            conv_2 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_1)
            dropout_2 = layers.Dropout(rate=self.configs["dropout"])(conv_2)
            batch_norm_2 = layers.BatchNormalization()(dropout_2)
            max_pool_2 = layers.MaxPooling2D((2, 2))(batch_norm_2)
            conv_3 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_2)
            dropout_3 = layers.Dropout(rate=self.configs["dropout"])(conv_3)
            batch_norm_3 = layers.BatchNormalization()(dropout_3)
            max_pool_3 = layers.MaxPooling2D((2, 2))(batch_norm_3)
    
            conv_4 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(batch_norm_3)
            dropout_4 = layers.Dropout(rate=self.configs["dropout"])(conv_4)
            batch_norm_4 = layers.BatchNormalization()(dropout_4)
            up_4 = layers.UpSampling2D((2, 2))(batch_norm_4)
            conv_5 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(up_4)
            dropout_5 = layers.Dropout(rate=self.configs["dropout"])(conv_5)
            batch_norm_5 = layers.BatchNormalization()(dropout_5)
            up_5 = layers.UpSampling2D((2, 2))(batch_norm_5)
            conv_6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal", padding="same")(up_5)
            conv_out = layers.Dropout(rate=self.configs["dropout"])(conv_6)

        # Version 3
        if self.configs["version"] == "v3":
            conv_1 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal", padding="same")(inp)
            dropout_1 = layers.Dropout(rate=self.configs["dropout"])(conv_1)
            batch_norm_1 = layers.BatchNormalization()(dropout_1)
            max_pool_1 = layers.MaxPooling2D((2, 2))(batch_norm_1)
            conv_2 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_1)
            dropout_2 = layers.Dropout(rate=self.configs["dropout"])(conv_2)
            batch_norm_2 = layers.BatchNormalization()(dropout_2)
            max_pool_2 = layers.MaxPooling2D((2, 2))(batch_norm_2)
            conv_3 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(max_pool_2)
            dropout_3 = layers.Dropout(rate=self.configs["dropout"])(conv_3)
            batch_norm_3 = layers.BatchNormalization()(dropout_3)
            max_pool_3 = layers.MaxPooling2D((2, 2))(batch_norm_3)

            conv_4 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(batch_norm_3)
            dropout_4 = layers.Dropout(rate=self.configs["dropout"])(conv_4)
            batch_norm_4 = layers.BatchNormalization()(dropout_4)
            up_4 = layers.UpSampling2D((2, 2))(batch_norm_4)
            conv_5 = layers.Conv2D(64, (3, 3), kernel_initializer="glorot_normal", activation='relu', padding="same")(up_4)
            dropout_5 = layers.Dropout(rate=self.configs["dropout"])(conv_5)
            batch_norm_5 = layers.BatchNormalization()(dropout_5) + batch_norm_2
            up_5 = layers.UpSampling2D((2, 2))(batch_norm_5)
            conv_6 = layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer="glorot_normal", padding="same")(up_5)
            conv_out = layers.Dropout(rate=self.configs["dropout"])(conv_6)

        flatten = layers.Flatten()(conv_out)
        dense_1 = layers.Dense(64, kernel_initializer="glorot_normal", activation='relu')(flatten)
        out = layers.Dense(self.configs["num_classes"], activation='softmax')(dense_1)

        model = Model(inputs=inp, outputs=out, name="cifar_10_model")

        return model


### END CODE HERE