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
    depth_major = tf.reshape(record, [3, 32, 32])
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
           image = tf.image.resize_image_with_crop_or_pad(image, 32 + 8, 32 + 8)

           # Randomly crop a [32, 32] section of the image.
           image = tf.random_crop(image, [32, 32, 3])

           # Randomly flip the image horizontally.
           image = tf.image.random_flip_left_right(image)
    # Subtract off the mean and divide by the standard deviation of the pixels.
    image = tf.image.per_image_standardization(image)
    ## END CODE HERE
    
    return image


# Other functions
### YOUR CODE HERE

### END CODE HERE