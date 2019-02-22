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
        # We perform pixel-wise normalization
        pixel_mean = self.X_train.mean(axis=0)
        pixel_std = self.X_train.std(axis=0) + 1e-10  # Prevent division-by-zero errors
        # Normalize the train and test set.
        self.X_train = (self.X_train - pixel_mean) / pixel_std
        self.X_test = (self.X_test - pixel_mean) / pixel_std
        self.X_train_vanilla = self.X_train

    def split_sets(self):
        # Only use the 20000 first samples in the training set
        self.X_train = self.X_train[:20000]
        self.Y_train = self.Y_train[:20000]
        self.X_train_vanilla = self.X_train
        self.Y_train_vanilla = self.Y_train

        # Only use the 2000 last samples in the test set
        self.X_test = self.X_test[8000:]
        self.Y_test = self.Y_test[8000:]

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
        weights = np.zeros((num_features, 10))
        learning_rate = lr
        num_batches_per_epoch = self.X_train.shape[0] // batch_size
        stop_epoch = epochs

        # Training loop. Train over a set amount of epochs with annealing learning rate and early stopping
        for epoch in range(epochs):
            print("epoch: {}/{}".format(epoch+1, epochs))
            # Train the network by dividing training data into batches for each epoch.
            # TODO: Might want to shuffle the training data.
            for i in range(num_batches_per_epoch):
                x_batch = self.X_train[i * batch_size:(i + 1) * batch_size]
                y_batch = self.Y_train[i * batch_size:(i + 1) * batch_size]

                targets = y_batch
                outputs = NN.forward_pass(x_batch, weights)

                weights = NN.gradient_decent(x_batch, outputs, targets, weights, learning_rate, batch_size)

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
            if (len(val_loss) > 5) and (min(val_loss[-5:]) == val_loss[-5]):
                stop_epoch = epoch + 1
                break

            # Annealing learning rate
            learning_rate = lr / (1+(epoch+1)/epochs)

        plt.figure(figsize=(12, 8))
        plt.plot(list(range(0, stop_epoch)), train_loss, label="Training loss")
        plt.plot(list(range(0, stop_epoch)), val_loss, label="Validation loss")
        plt.plot(list(range(0, stop_epoch)), test_loss, label="Test loss")
        plt.legend()
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.plot(list(range(0, stop_epoch)), train_accuracy, label="Training accuracy")
        plt.plot(list(range(0, stop_epoch)), val_accuracy, label="Validation accuracy")
        plt.plot(list(range(0, stop_epoch)), test_accuracy, label="Test accuracy")
        plt.legend()
        plt.show()

        print("Training accuracy: {}\n"
              "Validation accuracy: {}\n"
              "Test accuracy: {}".format(train_accuracy[-1], val_accuracy[-1], test_accuracy[-1]))

        # weight_zero = np.take(weights[:-1], indices=0, axis=1)
        # NN.draw_image([weight_zero, single_digit_set_mean], 0)

    @staticmethod
    def number_filter(x, y, num):
        indices_to_remove = [i for i, label in enumerate(y) if label != num]
        single_digit_set = np.delete(x, indices_to_remove, 0)
        single_digit_set_mean = single_digit_set.mean(axis=0)
        return single_digit_set_mean

    @staticmethod
    def draw_image(x_arr, num):
        for i, x in enumerate(x_arr):
            if i % 2 == 0:
                title = "Weights for number: " + str(num)
            else:
                title = "Average for dataset: " + str(num)
            img = np.reshape(x, (28, 28))
            plt.subplot(1, 2, i+1)
            plt.imshow(img, aspect='auto', cmap='gray')
            plt.title(title)
        plt.show()

    @staticmethod
    def cross_entropy(targets, outputs):
        eps = 1e-10
        assert targets.shape == outputs.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        error = - targets * np.log(outputs + eps)
        return error.mean()

    @staticmethod
    def gradient_decent(x, outputs, targets, weights, learning_rate, batch_size):
        # Important to check that your vectors/matrices have the expected shape
        assert outputs.shape == targets.shape, "outputs: {}, targets: {}".format(outputs.shape, targets.shape)
        dw = - x.transpose().dot(targets - outputs)
        dw /= (x.shape[0] + targets.shape[1])  # Normalize gradient w.r.t number of training samples
        dw += weights * 2 * 0.001
        assert dw.shape == weights.shape, "dw: {}, weights: {}".format(dw.shape, weights.shape)
        weights = weights - learning_rate * dw
        return weights

    @staticmethod
    def forward_pass(x, weights):
        return NN.softmax(x.dot(weights))

    @staticmethod
    def softmax(z):
        return np.exp(z) / np.sum(np.exp(z), axis=0)

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
    batch_size = 32
    learning_rate = 0.003

    nn = NN()
    nn.split_sets()
    nn.data_pre_processing()
    nn.trim_sets()
    nn.validation_split()
    # nn.statistics()
    nn.training(epochs, batch_size, learning_rate)


if __name__ == "__main__":
    main()
