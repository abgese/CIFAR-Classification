# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE
import tensorflow as tf

model_configs = {
	"name": 'MyModel',
	"input_size": (32, 32, 3),
	"num_classes": 10,
	"weight_decay": 1e-4,
	# ...
}

training_configs = {
	"optimizer": 'adam',
	"loss": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
	"batch_size": 32,
	"epochs": 100,
	# ...
}

### END CODE HERE