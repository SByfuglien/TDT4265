import mnist
import numpy as np
import matplotlib.pyplot as plt



class NN:

    def __init__(self):
        self.X_train, self.Y_train, self.X_test, self.Y_test = mnist.load()
        self.X_val = None
        self.Y_val = None
        self.X_train_vanilla = self.X_train
        self.Y_train_vanilla = self.Y_train

    def statistics(self):
        print("Number of training examples:", self.X_train.shape)
        print("Number of validation examples:", self.X_val.shape[0])
        print("Number of testing examples:", self.X_test.shape[0])
        print("Image shape:", self.X_train.shape)
        num_classes = np.unique(self.Y_train).shape[0]
        print("Number of classes:", num_classes)

    def data_pre_processing(self):
        # # We perform pixel-wise normalization
        # pixel_mean = self.X_train.mean(axis=0)
        # pixel_std = self.X_train.std(axis=0) + 1e-10  # Prevent division-by-zero errors
        # # Normalize the train and test set.
        # self.X_train = (self.X_train - pixel_mean) / pixel_std
        # self.X_test = (self.X_test - pixel_mean) / pixel_std

        self.X_train = (self.X_train / 127.5) - 1
        self.X_test = (self.X_test / 127.5) - 1

    def trim_sets(self):
        # The bias trick
        self.X_train = np.concatenate((self.X_train, np.ones((self.X_train.shape[0], 1))), axis=1)
        self.Y_train = NN.one_hot_encode(self.Y_train)
        self.X_test = np.concatenate((self.X_test, np.ones((self.X_test.shape[0], 1))), axis=1)
        self.Y_test = NN.one_hot_encode(self.Y_test)

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

    def training(self, epochs, batch_size, lr):
        train_loss = []
        val_loss = []
        test_loss = []
        train_accuracy = []
        val_accuracy = []
        test_accuracy = []
        num_features = self.X_train.shape[1]
        num_classes = self.Y_train.shape[1]
        weights = np.zeros((num_features, num_classes))
        learning_rate = lr
        num_batches_per_epoch = self.X_train.shape[0] // batch_size

        # Training loop. Train over a set amount of epochs with annealing learning rate and early stopping
        for epoch in range(epochs):
            # Train the network by dividing training data into batches for each epoch.
            # TODO: Shuffle the training data.
            for i in range(num_batches_per_epoch):
                x_batch = self.X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = self.Y_train[i * batch_size:(i + 1) * batch_size]

                targets = y_batch
                outputs = NN.forward_pass(x_batch, weights)

                weights = NN.gradient_decent(x_batch, outputs, targets, weights, learning_rate)

            # Loss and accuracy calculations on training set
            train_outputs = NN.forward_pass(self.X_train, weights)
            train_l = NN.cross_entropy(self.Y_train, train_outputs)
            train_loss.append(train_l)
            train_acc = NN.evaluation(train_outputs, self.Y_train)
            train_accuracy.append(train_acc)

            # Loss and accuracy calculations on validation set
            val_outputs = NN.forward_pass(self.X_val, weights)
            val_l = NN.cross_entropy(self.Y_val, val_outputs)
            val_loss.append(val_l)
            val_acc = NN.evaluation(val_outputs, self.Y_val)
            val_accuracy.append(val_acc)

            # Loss and accuracy calculations on test set
            test_outputs = NN.forward_pass(self.X_test, weights)
            test_l = NN.cross_entropy(self.Y_test, test_outputs)
            test_loss.append(test_l)
            test_acc = NN.evaluation(test_outputs, self.Y_test)
            test_accuracy.append(test_acc)

            # Early stopping
            if (len(val_loss) > 3) and (min(val_loss[-3:]) == val_loss[-3]):
                break

            # Annealing learning rate
            learning_rate = lr / (1+(epoch+1)/epochs)

            print("epoch: {}/{}".format(epoch+1, epochs))
            print("Training accuracy: {:04.2f}\n"
                  "Validation accuracy: {:04.2f}\n"
                  "Test accuracy: {:04.2f}\n".format(train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))

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

        print("Final training accuracy: {:04.2f}\n"
              "Final validation accuracy: {:04.2f}\n"
              "Final test accuracy: {:04.2f}".format(train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))

    @staticmethod
    def cross_entropy(targets, outputs):
        eps = 1e-10
        assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        error = - targets * np.log(outputs + eps)
        return error.mean()

    @staticmethod
    def gradient_decent(x, outputs, targets, weights, learning_rate):
        # Important to check that your vectors/matrices have the expected shape
        assert outputs.shape == targets.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        dw = - x.transpose().dot(targets - outputs)
        dw = dw / (x.shape[0] + targets.shape[1])  # Normalize gradient w.r.t number of training samples
        dw += weights * 2 * 0.001
        assert dw.shape == weights.shape, "dw: {}, weights: {}".format(dw.shape, weights.shape)
        weights = weights - learning_rate * dw
        return weights

    @staticmethod
    def forward_pass(x, weights):
        return NN.softmax(x.dot(weights))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    @staticmethod
    def evaluation(outputs, targets):
        assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        outputs = outputs.argmax(axis=1)
        targets = targets.argmax(axis=1)
        correct = np.sum(outputs == targets)
        return 100 * correct / len(outputs)

    @staticmethod
    def one_hot_encode(batch):
        res = np.eye(10)[np.array(batch).reshape(-1)]
        return res.reshape(list(batch.shape) + [10])


def main():
    # Hyperparameters
    epochs = 50
    batch_size = 128
    learning_rate = 0.003

    nn = NN()
    nn.data_pre_processing()
    nn.trim_sets()
    nn.validation_split()
    # nn.statistics()
    nn.training(epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
