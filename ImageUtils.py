import numpy as np
import tensorflow as tf

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    depth_major = record.reshape((3, 32, 32))
    image = np.transpose(depth_major, [1, 2, 0])
    ### END CODE HERE

    image = preprocess_image(image, training) # If any.

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
           # Resize the image to add four extra pixels on each side.
           image = np.pad(image, ((4,4),(4,4),(0,0)), 'constant')

           # Randomly crop a [32, 32] section of the image.
           upper_x = np.random.randint(0,8)
           upper_y = np.random.randint(0,8)
           image = image[upper_x:upper_x+32, upper_y:upper_y+32, :]

           # Randomly flip the image horizontally.
           coin_flip = np.random.randint(0,2)
           if (coin_flip == 0):
               image = np.fliplr(image)
    # Subtract off the mean and divide by the standard deviation of the pixels.
    mean = np.mean(image, axis=(0,1), keepdims=True)
    std = np.mean(image, axis=(0,1), keepdims=True)
    image = (image - mean)/std
    ## END CODE HERE
    
    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE