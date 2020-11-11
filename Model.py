### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, time
import numpy as np
from Network import MyNetwork
from ImageUtils import parse_record

"""This script defines the training, validation and testing process.
"""

class MyModel(object):

    def __init__(self, model_configs, training_configs):
        self.model_configs = model_configs
        self.training_configs = training_configs
        self.network = MyNetwork(model_configs)
        self.model = self.network()

    def model_setup(self):
        self.model.compile(
            optimizer = Adam(
                lr=self.training_configs["learning_rate"], 
                beta_1=self.training_configs["beta_1"], 
                beta_2=self.training_configs["beta_2"], 
                epsilon=self.training_configs["epsilon"], 
                decay=self.training_configs["decay"], 
                amsgrad=True), 
            loss = self.training_configs["loss"], 
            metrics = ['accuracy'])

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        if x_valid:
            validation_data = (x_val, y_val)
        else:
            validation_data = None
        self.model.fit(
            x_train, 
            y_train, 
            batch_size=self.training_configs["batch_size"], 
            epochs=self.training_configs["epochs"], 
            validation_data=validation_data)

    def evaluate(self, x, y):
        accuracy = self.model.evaluate(x, y, batch_size=self.training_configs["batch_size"])
        print("Model Accuracy : {}", accuracy)

    def predict_prob(self, x):
        return self.model.predict(x)

    def save_weights(self, path):
        return self.model.save_weights(path)

    def load_weights(self, pretrained_weights):
        self.model.load_weights(pretrained_weights)


### END CODE HERE