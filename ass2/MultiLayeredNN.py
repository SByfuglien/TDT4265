import mnist
import numpy as np
import matplotlib.pyplot as plt
import copy


class NN:

	def __init__(self, layers):
		"""Initializes the network. Loads training and test data."""
		self.X_train, self.Y_train, self.X_test, self.Y_test = mnist.load()
		self.X_val = None
		self.Y_val = None
		self.X_train_vanilla = self.X_train
		self.Y_train_vanilla = self.Y_train
		self.num_layers = len(layers)
		self.weights = self.initialize_improved_weights(layers)

	@staticmethod
	def initialize_weights(layers):
		"""Uniform weight initialization between -1 and 1."""
		weights = []
		for i in range(len(layers) - 1):
			weights.append(np.random.uniform(-1, 1, (layers[i], layers[i + 1])))
		return np.array(weights)

	@staticmethod
	def initialize_improved_weights(layers):
		"""Weight initialization"""
		weights = [np.random.normal(size=(x, y), scale=(1 / np.sqrt(x))) for x, y in zip(layers[:-1], layers[1:])]
		return np.array(weights)

	def statistics(self):
		"""Statics for data set. For illustration purposes"""
		print("Number of training examples:", self.X_train.shape)
		print("Number of validation examples:", self.X_val.shape[0])
		print("Number of testing examples:", self.X_test.shape[0])
		print("Image shape:", self.X_train.shape)
		num_classes = np.unique(self.Y_train).shape[0]
		print("Number of classes:", num_classes)

	def data_pre_processing(self):
		"""Normalizing input. Point (b) task 2. """
		self.X_train = (self.X_train / 127.5) - 1
		self.X_test = (self.X_test / 127.5) - 1

	def trim_sets(self):
		"""Performs the bias trick on training and test set to simplify subsequent operations.
		Also one-hot encodes the target sets. Part of solution for (a) in task 2."""
		# The bias trick
		self.X_train = np.concatenate((self.X_train, np.ones((self.X_train.shape[0], 1))), axis=1)
		self.X_test = np.concatenate((self.X_test, np.ones((self.X_test.shape[0], 1))), axis=1)

		# One hot encoding
		self.Y_train = NN.one_hot_encode(self.Y_train)
		self.Y_test = NN.one_hot_encode(self.Y_test)

	def validation_split(self):
		"""Split training set into validation and training set. Part of solution for  (a) in task 2."""
		# We want to use a validation set to validate our model, and keep the test set for testing purposes only.
		train_percentage = 0.9  # Percentage of data to use in training set.
		indexes = np.arange(self.X_train.shape[0])
		np.random.shuffle(indexes)
		# Select random indexes for train/val set.
		idx_train = indexes[:int(train_percentage * self.X_train.shape[0])]
		idx_val = indexes[int(train_percentage * self.X_train.shape[0]):]

		self.X_val = self.X_train[idx_val]
		self.Y_val = self.Y_train[idx_val]

		self.X_train = self.X_train[idx_train]
		self.Y_train = self.Y_train[idx_train]

	def training(self, epochs, batch_size, lr, momentum):
		"""Method for training the network."""
		train_loss = []
		val_loss = []
		test_loss = []
		train_accuracy = []
		val_accuracy = []
		test_accuracy = []

		learning_rate = lr
		num_batches_per_epoch = self.X_train.shape[0] // batch_size
		prev_gradient = 0

		# Training loop. Train over a set amount of epochs.
		for epoch in range(epochs):
			# Train the network by dividing training data into batches for each epoch.
			# Shuffle the trainings set for each epoch.
			self.X_train, self.Y_train = self.shuffle(self.X_train, self.Y_train)
			for i in range(num_batches_per_epoch):
				x_batch = self.X_train[i * batch_size:(i + 1) * batch_size]
				y_batch = self.Y_train[i * batch_size:(i + 1) * batch_size]

				targets = y_batch
				outputs = NN.forward_pass(x_batch, self.weights)

				self.weights, prev_gradient = NN.gradient_descent(x_batch, outputs, targets, self.weights,
																  learning_rate, prev_gradient, momentum)

			# Loss and accuracy calculations on training set. Performed after each epoch.
			train_outputs = NN.forward_pass(self.X_train, self.weights)
			train_l = NN.cross_entropy(self.Y_train, train_outputs)
			train_loss.append(train_l)
			train_acc = NN.evaluation(train_outputs, self.Y_train)
			train_accuracy.append(train_acc)

			# Loss and accuracy calculations on validation set.
			val_outputs = NN.forward_pass(self.X_val, self.weights)
			val_l = NN.cross_entropy(self.Y_val, val_outputs)
			val_loss.append(val_l)
			val_acc = NN.evaluation(val_outputs, self.Y_val)
			val_accuracy.append(val_acc)

			# Loss and accuracy calculations on test set.
			test_outputs = NN.forward_pass(self.X_test, self.weights)
			test_l = NN.cross_entropy(self.Y_test, test_outputs)
			test_loss.append(test_l)
			test_acc = NN.evaluation(test_outputs, self.Y_test)
			test_accuracy.append(test_acc)

			# Early stopping
			if (len(val_loss) > 5) and (min(val_loss[-5:]) == val_loss[-5]):
				break

			# Annealing learning rate
			learning_rate = lr / (1 + epoch / epochs)

			print("epoch: {}/{}".format(epoch + 1, epochs))
			print("Training accuracy: {:04.2f}\nValidation accuracy: {:04.2f}\nTest accuracy: {:04.2f}\n".format(
				train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))

		return train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy

	@staticmethod
	def shuffle(x, y):
		"""Shuffle the training set"""
		indexes = np.arange(x.shape[0])
		np.random.shuffle(indexes)
		x = x[indexes]
		y = y[indexes]
		return x, y

	@staticmethod
	def plots(train_loss, val_loss, test_loss, train_accuracy, val_accuracy, test_accuracy):
		"""Method for plotting of accuracy and loss."""
		plt.figure(figsize=(12, 8))
		plt.plot(train_loss, label="Training loss")
		plt.plot(val_loss, label="Validation loss")
		plt.plot(test_loss, label="Test loss")
		plt.legend()
		plt.show()

		plt.figure(figsize=(12, 8))
		plt.plot(train_accuracy, label="Training accuracy")
		plt.plot(val_accuracy, label="Validation accuracy")
		plt.plot(test_accuracy, label="Test accuracy")
		plt.legend()
		plt.show()

		print("Final training accuracy: {:04.2f}\n""Final validation accuracy: {:04.2f}\n"
			  "Final test accuracy: {:04.2f}".format(train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))

	@staticmethod
	def cross_entropy(outputs, targets):
		"""Calculates loss given outputs(predictions by the network) and targets(true values for input)."""
		eps = 1e-10
		assert outputs.shape == targets.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
		# Cost function for multi-class classification
		error = - targets * np.log(outputs + eps)
		return error.mean()

	@staticmethod
	def gradient_descent(x, outputs, targets, weights, learning_rate, prev_gradient, momentum):
		"""Gradient descent method."""
		# Important to check that your vectors/matrices have the expected shape.
		assert outputs.shape == targets.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
		# Perform backpropagation to find gradients.
		gradient_w = NN.back_propagation(x, targets, weights)
		# NN.check_gradient(x, targets, weights, 1, gradient_w[1])
		assert gradient_w.shape == weights.shape, "dw: {}, weights: {}".format(gradient_w.shape, weights.shape)
		# L2 regularization
		gradient_w += weights * 2 * 0.001
		# Use momentum to update weights.
		dw = NN.momentum(learning_rate, momentum, gradient_w, prev_gradient)
		weights = weights - dw
		return weights, dw

	@staticmethod
	def back_propagation(x, targets, weights):
		"""Backpropagation algorithm for gradient descent. Performs forward pass through the network
		before performing gradient calculations (the backpropagation)."""
		normalization_factor = x.shape[0] * targets.shape[1]
		a = x
		activations = [x]
		weighted_inputs = []
		gradient_w = [np.zeros(w.shape) for w in weights]
		# Forward pass
		for layer in range(len(weights)):
			z = a.dot(weights[layer])
			if layer == len(weights) - 1:
				a = NN.softmax(z)
			else:
				a = NN.relu(z)
			activations.append(a)
			weighted_inputs.append(z)
		# Gradient calculation for output layer
		delta_k = - np.subtract(targets, activations[-1])
		gradient_w[-1] = np.dot(delta_k.transpose(), activations[-2]).transpose()
		delta_j = delta_k
		# Gradient calculation for hidden layers
		for layer in range(2, len(weights) + 1):
			z = weighted_inputs[-layer]
			ds_dz = NN.relu_derivative(z)
			delta_j = np.dot(delta_j, weights[-layer + 1].transpose())
			delta_j *= ds_dz
			dw = np.dot(delta_j.transpose(), activations[-layer - 1]).transpose()
			gradient_w[-layer] = dw
		# for gradient in gradient_w:
		# 	gradient /= normalization_factor
		return np.array(gradient_w)

	@staticmethod
	def check_gradient(x, targets, weights, layer, computed_gradient, epsilon=1e-2):
		"""Method for numerically checking the gradient calculations using finite differences.
		Used for debugging purposes. Shamelessly stolen from example code. Solution for (d) in task 2."""
		print("Checking gradient...")
		dw = np.zeros_like(weights[layer])
		for k in range(weights[layer].shape[0]):
			for j in range(weights[layer].shape[1]):
				new_weight1, new_weight2 = copy.deepcopy(weights), copy.deepcopy(weights)
				new_weight1[layer][k, j] += epsilon
				outputs1 = NN.forward_pass(x, new_weight1)
				loss1 = NN.cross_entropy(outputs1, targets)
				new_weight2[layer][k, j] -= epsilon
				outputs2 = NN.forward_pass(x, new_weight2)
				loss2 = NN.cross_entropy(outputs2, targets)
				dw[k, j] = (loss1 - loss2) / (2 * epsilon)

		maximum_absolute_difference = abs(computed_gradient - dw).max()
		assert maximum_absolute_difference <= epsilon ** 2, "Absolute error was: {}, Epsilon squared: {}".format(
			maximum_absolute_difference, epsilon ** 2)

	@staticmethod
	def forward_pass(x, weights):
		"""Forward pass through the network. Converting inputs to outputs. Using Sigmoid function for hidden layers,
		and Softmax for output layer."""
		a = x
		for layer in range(len(weights)):
			z = a.dot(weights[layer])
			if layer == len(weights) - 1:
				a = NN.softmax(z)
			else:
				a = NN.relu(z)
		return a

	@staticmethod
	def softmax(z):
		"""Softmax activation function for output layer. Solution for (c) in task 2."""
		return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

	@staticmethod
	def sigmoid(z):
		"""Sigmoid activation function for hidden layer(s). Solution for (c) in task 2."""
		return 1.0 / (1.0 + np.exp(-z))

	@staticmethod
	def sigmoid_derivative(z):
		"""Derivative of Sigmoid activation function for hidden layer(s). To be used in backpropagation"""
		a = NN.sigmoid(z)
		return a * (1 - a)

	@staticmethod
	def evaluation(outputs, targets):
		"""Method for determining number of correct outputs from the network."""
		assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
		outputs = outputs.argmax(axis=1)
		targets = targets.argmax(axis=1)
		correct = np.sum(outputs == targets)
		return 100 * correct / len(outputs)

	@staticmethod
	def one_hot_encode(y, n_classes=10):
		"""One hot encoding. Important for multi-class classification problems."""
		one_hot = np.zeros((y.shape[0], n_classes))
		one_hot[np.arange(0, y.shape[0]), y] = 1
		return one_hot

	@staticmethod
	def improved_sigmoid(z):
		"""Improved sigmoid activation function, Task 3b"""
		return 1.7159 * np.tanh((2 / 3) * z)

	@staticmethod
	def improved_sigmoid_derivative(z):
		"""Derivative of mproved sigmoid activation function, Task 3b"""
		return 1.7159 * (2 / (3 * (np.cosh((2 * z) / 3)) ** 2))

	@staticmethod
	def relu(z):
		return np.maximum(0.0, z)

	@staticmethod
	def relu_derivative(z):
		return 1. * (z > 0)

	@staticmethod
	def momentum(lr, momentum, gradient, prev_gradient):
		"""Momentum implementation,Task 3d"""
		return gradient * lr + prev_gradient * momentum


# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.0003
MOMENTUM = 0.9

# Pass in neurons for each layer as a list, 785 for the first and 10 for the last layer.
nn = NN([785, 256, 128, 10])
nn.data_pre_processing()
nn.trim_sets()
nn.validation_split()
TRAIN_LOSS, VAL_LOSS, TEST_LOSS, TRAIN_ACCURACY, VAL_ACCURACY, TEST_ACCURACY = nn.training(EPOCHS, BATCH_SIZE,
																						   LEARNING_RATE, MOMENTUM)
nn.plots(TRAIN_LOSS, VAL_LOSS, TEST_LOSS, TRAIN_ACCURACY, VAL_ACCURACY, TEST_ACCURACY)
