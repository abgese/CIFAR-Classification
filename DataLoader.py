import os
import pickle
import numpy as np
from ImageUtils import parse_record
"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    X = np.array([])
    Y = np.array([])
    first = True
    for file_name in os.listdir(data_dir):
    	with open(os.path.join(data_dir, file_name), 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
            if(first):
                X = np.array(data_dict[b'data'])
                Y = np.array(data_dict[b'labels'])
                first = False
            else:
                X = np.append(X, np.array(data_dict[b'data']), axis = 0)
                Y = np.append(Y, np.array(data_dict[b'labels']))
    ### END CODE HERE
    X_parsed = []
    for i in range(len(X)):
        X_parsed.append(parse_record(X[i], True))
    X_parsed = np.array(X_parsed)
    x_train = X_parsed[:50000]
    y_train = Y[:50000]
    x_test = X_parsed[50000:]
    y_test = Y[50000:]
    return x_train, y_train, x_test, y_test

def load_testing_images(data_file):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    X = np.load(data_file)
    x_test = []
    for x in X:
        x_test.append(parse_record(x, False))
    x_test = np.array(x_test)
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = int(len(x_train) * train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

