# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
import tensorflow as tf
from tensorflow.keras import optimizers

model_configs = {
	"name": 'MyModel',
	"input_size": (32, 32, 3),
	"num_classes": 10,
	"dropout": 0.2,
	"version": "v2"
	# ...
}

training_configs = {
	"optimizer": optimizers.Adam(lr=0.00005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True),
	"loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	"batch_size": 32,
	"epochs": 300,
	# ...
}

### END CODE HERE