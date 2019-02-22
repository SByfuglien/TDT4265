import keras
from keras.datasets import cifar10
from keras.models import Sequential
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sn
import pandas as pd


class CNN:

	def __init__(self):
		# Initialize a convolutional neural network.
		(self.X_train, self.Y_train),(self.X_test, self.Y_test) = cifar10.load_data()
		self.num_classes = np.unique(self.Y_train).shape[0]
		# Boolean used for determining the correct reshaping of training and test sets.
		self.X_val = None
		self.Y_val = None
		self.model = Sequential()
		# Used to visualize images in the analysis method
		self.vanilla_X_test = self.X_test

	def visualize_dataset(self, visualize_bool, statistics_bool, class_distribution_bool):
		# Visualize some images
		if visualize_bool:
			for row in range(3):
				for col in range(3):
					idx = row * 3 + col + 1
					plt.subplot(3, 3, idx)
					plt.imshow(self.X_train[idx - 1], cmap="gray")

		# Basic dataset Statistics
		if statistics_bool:
			print("Number of training examples:", self.X_train.shape[0])
			print("Number of testing examples:", self.X_test.shape[0])
			print("Number of classes:", self.num_classes)
			print("Image shape:", self.X_train[0].shape)
			print("Image data type:", self.X_train.dtype)

		# Plot class distribution
		if class_distribution_bool:
			class_distribution = Counter(self.Y_train[:, 0])
			x = range(10)
			y = [class_distribution[cls] for cls in x]
			plt.figure(figsize=(12, 8))
			plt.xticks(x)
			plt.title("Number of training examples in each class")
			plt.xlabel("Class")
			plt.ylabel("Number of examples")
			plt.bar(x, y)

	def data_pre_processing(self):
		# We perform pixel-wise normalization
		pixel_mean = self.X_train.mean(axis=0)
		pixel_std = self.X_train.std(axis=0) + 1e-10  # Prevent division-by-zero errors
		# Normalize the train and test set.
		# (This is also done in each convolutional layer by batch normalization, so not really necessary)
		self.X_train = (self.X_train - pixel_mean) / pixel_std
		self.X_test = (self.X_test - pixel_mean) / pixel_std

	def change_data_shape(self):
		self.Y_train = keras.utils.to_categorical(self.Y_train, self.num_classes)
		self.Y_test = keras.utils.to_categorical(self.Y_test, self.num_classes)

		self.X_train = self.X_train.reshape(self.X_train.shape[0], 32, 32, 3)
		self.X_test = self.X_test.reshape(self.X_test.shape[0], 32, 32, 3)

	def split_dataset(self):
		# We want to use a validation set to validate our model, and keep the test set for testing purposes only.
		train_val_split = 0.9  # Percentage of data to use in training set
		indexes = np.arange(self.X_train.shape[0])
		np.random.shuffle(indexes)
		# Select random indexes for train/val set
		idx_train = indexes[:int(train_val_split * self.X_train.shape[0])]
		idx_val = indexes[int(train_val_split * self.X_train.shape[0]):]

		self.X_val = self.X_train[idx_val]
		self.Y_val = self.Y_train[idx_val]

		self.X_train = self.X_train[idx_train]
		self.Y_train = self.Y_train[idx_train]

		# print("Training set shape:", self.X_train.shape)
		# print("Validation set shape:", self.X_val.shape)
		# print("Testing set shape:", self.X_test.shape)

	def model_train(self, num_epochs, batch_size, data_aug=False):
		# Using real-time data augmentation to train the model.
		if data_aug:
			data_gen = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
																	horizontal_flip=True,
																	width_shift_range=0.1,
																	height_shift_range=0.1)
			data_gen.fit(self.X_train)
			self.model.fit_generator(data_gen.flow(self.X_train, self.Y_train, batch_size=batch_size),
									 steps_per_epoch=len(self.X_train) / batch_size, epochs=num_epochs,
									 validation_data=(self.X_val, self.Y_val), verbose=2)
		# Training the model the usual way
		else:
			self.model.fit(self.X_train, self.Y_train,
						   batch_size=batch_size,
						   epochs=num_epochs,
						   verbose=2,
						   validation_data=(self.X_val, self.Y_val))

	def model_analysis(self, loss_bool=False, acc_bool=True, comp_bool=True):
		# Loss history
		history = self.model.history.history
		if loss_bool:
			plt.figure(figsize=(12, 8))
			plt.plot(history["val_loss"], label="Validation loss")
			plt.plot(history["loss"], label="Training loss")
			plt.legend()

		# Accuracy history
		if acc_bool:
			plt.figure(figsize=(12, 8))
			plt.plot(history["val_acc"], label="Validation accuracy")
			plt.plot(history["acc"], label="Training accuracy")
			plt.legend()

		# True and predicted classifications for the model
		true = np.argmax(self.Y_test, axis=1)
		pred = np.argmax(self.model.predict(self.X_test), axis=1)

		# List of indices where the model misclassifies images
		fails = [i for i, item in enumerate(true) if item != pred[i]]

		print("num of fails: ", len(fails))

		# Visualizing a few of the misclassifications by the network
		for i in range(9):
			ind = fails[i]
			plt.subplot(3, 3, i+1)
			plt.subplots_adjust(hspace=0.5)
			plt.imshow(self.vanilla_X_test[ind], cmap="gray")
			plt.title("pred: {}, true = {}".format(pred[ind], true[ind]))

		# Computing the confusion matrix and f1 score
		cm = metrics.confusion_matrix(true, pred)
		f1 = metrics.f1_score(true, pred, average=None)

		# print('f1 score: ', f1)
		# recall = metrics.recall_score(true, pred, average=None)
		# print('recall score: ', recall)
		# precision = metrics.precision_score(true, pred, average=None)
		# print('precision score: ', precision)

		# Final evaluation on test set
		final_loss, final_accuracy = self.model.evaluate(self.X_test, self.Y_test)
		print("The final loss on the test set is:", final_loss)
		print("The final accuracy on the test set is:", final_accuracy)

		# Plotting f1 scores for each class
		x = range(10)
		plt.figure(figsize=(12, 8))
		plt.xticks(x)
		plt.title("F1 score for each class")
		plt.xlabel("Class")
		plt.ylabel("F1 score")
		plt.bar(x, f1)

		# Using labraries to plot the confusion matrix
		df_cm = pd.DataFrame(cm, index=range(10),
							 columns=range(10))
		plt.figure(figsize=(10, 7))
		sn.heatmap(df_cm, annot=True, fmt='.5g')

		# Compare the test loss on our previous plot.
		history = self.model.history.history
		if comp_bool:
			plt.figure(figsize=(12, 8))
			plt.plot(history["val_loss"], label="Validation loss")
			plt.plot(history["loss"], label="Training loss")
			plt.plot([9], [final_loss], 'o', label="Final test loss")
			plt.legend()
		return final_accuracy

	def setup(self, visualize_bool=False, statistics_bool=False, class_distribution_bool=False):
		self.visualize_dataset(visualize_bool, statistics_bool, class_distribution_bool)
		self.data_pre_processing()
		self.change_data_shape()
		self.split_dataset()
