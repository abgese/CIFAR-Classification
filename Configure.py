# Below configures are examples, 
# you can modify them as you wish.

### YOUR CODE HERE

model_configs = {
	"name": 'MyModel',
	"save_dir": '../saved_models/',
	"depth": 2,
	"input_size": (32, 32, 3),
	"num_classes": 10,
	# ...
}

training_configs = {
	"learning_rate": 0.01,
	"epsilon": 1e-08,
	"decay": 0.0,
	"beta_1": 0.9,
	"beta_2": 0.999,
	"loss": "categorical_crossentropy",
	"batch_size": 64,
	"epochs": 100,
	# ...
}

### END CODE HERE