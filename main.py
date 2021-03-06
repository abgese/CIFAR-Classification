### YOUR CODE HERE
# import tensorflow as tf
# import torch
import os, argparse
import numpy as np
from Model import MyModel
from DataLoader import load_data, train_valid_split, load_testing_images
from Configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("--mode", help="train, test or predict")
parser.add_argument("--data_dir", help="path to the data")
parser.add_argument("--test_file", help="path to the test file")
parser.add_argument("--save_dir", help="path to save the results")
args = parser.parse_args()

if __name__ == '__main__':
	model = MyModel(model_configs, training_configs)

	if args.mode == 'train':
		x_train, y_train , x_test, y_test = load_data(args.data_dir)
		x_train, y_train, x_valid, y_valid = train_valid_split(x_train, y_train)

		model.train(x_train, y_train, x_valid, y_valid)
		model.save_weights(os.path.join(args.save_dir, model_configs["version"], "") )
		model.evaluate(x_test, y_test)

	elif args.mode == 'test':
		# Testing on public testing dataset
		model.load_weights(os.path.join(args.save_dir, model_configs["version"], "") )
		_, _, x_test, y_test = load_data(args.data_dir)
		model.evaluate(x_test, y_test)

	elif args.mode == 'predict':
		# Predicting and storing results on private testing dataset 
		model.load_weights(os.path.join(args.save_dir, model_configs["version"], ""))
		x_test = load_testing_images(args.test_file)
		predictions = model.predict_prob(x_test)
		np.save("final_pred_"+ model_configs["version"] + ".npy", predictions)

		

### END CODE HERE

