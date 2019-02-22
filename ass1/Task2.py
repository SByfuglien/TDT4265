import mnist
import numpy as np
import matplotlib.pyplot as plt


class NN:

    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = mnist.load()
        self.X_val = None
        self.Y_val = None

    def statistics(self):
        print("Number of training examples:", self.X_train.shape[0])
        print("Number of validation examples:", self.X_val.shape[0])
        print("Number of testing examples:", self.X_test.shape[0])
        print("Image shape:", self.Y_train.shape)
        num_classes = np.unique(self.Y_train).shape[0]
        print("Number of classes:", num_classes)

    def data_pre_processing(self):
        # We perform pixel-wise normalization
        pixel_mean = self.X_train.mean(axis=0)
        pixel_std = self.X_train.std(axis=0) + 1e-10  # Prevent division-by-zero errors
        # Normalize the train and test set.
        # (This is also done in each convolutional layer by batch normalization, so not really necessary)
        self.X_train = (self.X_train - pixel_mean) / pixel_std
        self.X_test = (self.X_test - pixel_mean) / pixel_std

    def split_sets(self):
        # Only use the 20000 first samples in the training set
        self.X_train = self.X_train[:20000]
        self.Y_train = self.Y_train[:20000]

        # Only use the 2000 last samples in the test set
        self.X_test = self.X_test[8000:]
        self.Y_test = self.Y_test[8000:]

    def trim_sets(self):
        # Removing images that are not 2 or 3's
        indices_to_remove = [i for i, label in enumerate(self.Y_train) if label != 2 and label != 3]
        self.Y_train = np.delete(self.Y_train, indices_to_remove, 0)
        self.X_train = np.delete(self.X_train, indices_to_remove, 0)

        indices_to_remove = [i for i, label in enumerate(self.Y_test) if label != 2 and label != 3]
        self.Y_test = np.delete(self.Y_test, indices_to_remove, 0)
        self.X_test = np.delete(self.X_test, indices_to_remove, 0)

        # The bias trick
        self.X_train = np.concatenate((self.X_train, np.ones((self.X_train.shape[0], 1))), axis=1)
        self.Y_train = self.Y_train.reshape(self.Y_train.shape[0], 1)
        self.X_test = np.concatenate((self.X_test, np.ones((self.X_test.shape[0], 1))), axis=1)
        self.Y_test = self.Y_test.reshape(self.Y_test.shape[0], 1)

    def validation_split(self):
        # We want to use a validation set to validate our model, and keep the test set for testing purposes only.
        train_percentage = 0.9  # Percentage of data to use in training set
        indexes = np.arange(self.X_train.shape[0])
        np.random.shuffle(indexes)
        # Select random indexes for train/val set
        idx_train = indexes[:int(train_percentage * self.X_train.shape[0])]
        idx_val = indexes[int(train_percentage * self.X_train.shape[0]):]

        self.X_val = self.X_train[idx_val]
        self.Y_val = self.Y_train[idx_val]

        self.X_train = self.X_train[idx_train]
        self.Y_train = self.Y_train[idx_train]

        # print("Training set shape:", self.X_train.shape)
        # print("Validation set shape:", self.X_val.shape)
        # print("Testing set shape:", self.X_test.shape)

    def training(self, epochs, batch_size, lr, lb):
        train_loss = []
        val_loss = []
        test_loss = []
        train_accuracy = []
        val_accuracy = [[], [], []]
        test_accuracy = []
        num_features = self.X_train.shape[1]
        weights = np.zeros((num_features, 1))
        weights_arr = [weights, weights, weights]
        learning_rate = lr
        num_batches_per_epoch = self.X_train.shape[0] // batch_size
        stop_epoch = epochs

        val_outputs = [[], [], []]

        # Training loop. Train over a set amount of epochs with annealing learning rate and early stopping
        for epoch in range(epochs):
            # Train the network by dividing training data into batches for each epoch.
            # TODO: Might want to shuffle the training data.
            for i in range(num_batches_per_epoch):
                x_batch = self.X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = self.Y_train[i * batch_size:(i + 1) * batch_size]
                targets = (y_batch == 2).astype(np.int)
                # print(y_batch.shape)

                for j, l in enumerate(lb):
                    outputs = NN.forward_pass(x_batch, weights_arr[j])
                    weights_arr[j] = NN.gradient_decent(x_batch, outputs, targets, weights_arr[j], learning_rate, l)

            # Loss and accuracy calculations on training set
            # train_outputs = NN.forward_pass(self.X_train, weights)
            # train_targets = (self.Y_train == 2).astype(np.int)
            # train_l = NN.binary_cross_entropy(train_targets, train_outputs)
            # train_loss.append(train_l)
            # train_acc = NN.binary_evaluation(train_outputs, train_targets)
            # train_accuracy.append(train_acc)

            # Loss and accuracy calculations on validation set
            for i, weight in enumerate(weights_arr):
                val_outputs[i] = NN.forward_pass(self.X_val, weight)
            val_targets = (self.Y_val == 2).astype(np.int)
            # val_l = NN.binary_cross_entropy(val_targets, val_outputs)
            # val_loss.append(val_l)
            for i, val_output in enumerate(val_outputs):
                val_acc = NN.binary_evaluation(val_output, val_targets)
                val_accuracy[i].append(val_acc)

            # Loss and accuracy calculations on test set
            # test_outputs = NN.forward_pass(self.X_test, weights)
            # test_targets = (self.Y_test == 2).astype(np.int)
            # test_l = NN.binary_cross_entropy(test_targets, test_outputs)
            # test_loss.append(test_l)
            # test_acc = NN.binary_evaluation(test_outputs, test_targets)
            # test_accuracy.append(test_acc)

            # Early stopping
            if (len(val_loss) > 5) and (min(val_loss[-5:]) == val_loss[-5]):
                stop_epoch = epoch+1
                break

            # Annealing learning rate
            learning_rate = lr / (1+(epoch+1)/epochs)

        # plt.figure(figsize=(12, 8))
        # plt.plot(list(range(0, stop_epoch)), train_loss, label="Training loss")
        # plt.plot(list(range(0, stop_epoch)), val_loss, label="Validation loss")
        # plt.plot(list(range(0, stop_epoch)), test_loss, label="Test loss")
        # plt.legend()  # Shows graph labels
        # plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(list(range(0, stop_epoch)), val_accuracy[0], label="Training accuracy")
        plt.plot(list(range(0, stop_epoch)), val_accuracy[1], label="Validation accuracy")
        plt.plot(list(range(0, stop_epoch)), val_accuracy[2], label="Test accuracy")
        plt.legend()  # Shows graph labels
        plt.show()

        print("Training accuracy: {}\n"
              "Validation accuracy: {}\n"
              "Test accuracy: {}".format(val_accuracy[0][-1], val_accuracy[1][-1], val_accuracy[2][-1]))

    @staticmethod
    def binary_cross_entropy(targets, outputs):
        eps = 1e-10
        assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        error = - targets * np.log(outputs + eps) - (1 - targets) * np.log(1 - outputs + eps)
        return error.mean()

    @staticmethod
    def gradient_decent(x, outputs, targets, weights, learning_rate, lb):
        # Important to check that your vectors/matrices have the expected shape
        assert outputs.shape == targets.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        # print("x: {}, outputs: {}, targets: {}".format(x.shape, outputs.shape, targets.shape))

        dw = x * (targets - outputs)
        # print("dw: {}, weights: {}".format(dw.shape, weights.shape))
        dw = dw.mean(axis=0).reshape(-1, 1)  # Normalize gradient w.r.t number of training samples
        dw += weights * 2 * lb
        assert dw.shape == weights.shape, "dw: {}, weights: {}".format(dw.shape, weights.shape)
        # print("dw: {}, weights: {}".format(dw.shape, weights.shape))
        weights = weights + learning_rate * dw
        return weights

    @staticmethod
    def forward_pass(x, weights):
        return NN.sigmoid(x.dot(weights))

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

    @staticmethod
    def binary_evaluation(outputs, targets):
        assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        outputs = (outputs >= 0.5).astype(np.int)
        correct = np.sum(outputs == targets)
        return 100 * correct / len(outputs)


def main():
    # Hyperparameters
    epochs = 50
    batch_size = 32
    learning_rate = 0.003
    lb = [0.01, 0.001, 0.0001]

    nn = NN()
    nn.data_pre_processing()
    nn.split_sets()
    nn.trim_sets()
    nn.validation_split()
    # nn.statistics()
    nn.training(epochs, batch_size, learning_rate, lb)


if __name__ == "__main__":
    main()
