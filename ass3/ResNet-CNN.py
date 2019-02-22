from CNN import CNN
import keras
from keras.layers import Dense, Flatten
from keras.optimizers import SGD
from keras_applications import resnet50

import time


class ResNetCNN(CNN):

    def __init__(self):
        super().__init__()

    def model_construction(self, learning_rate):
        resnet = resnet50.ResNet50(include_top=False,
                          weights='imagenet',
                          input_shape=self.X_train.shape[1:])
        for layer in resnet.layers[:-5]:
            layer.trainable = False

        self.model.add(resnet)

        self.model.add(Flatten())

        self.model.add(Dense(10, activation="softmax"))

        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.summary()


def main():
    start = time.time()
    learning_rate = 0.003
    num_epochs = 10
    batch_size = 64
    cnn = ResNetCNN()
    cnn.setup()
    cnn.model_construction(learning_rate)
    cnn.model_train(num_epochs, batch_size, data_aug=True)
    # cnn.model_analysis()
    cnn.model.save('models/ResNet-CNN')

    end = time.time()
    print("elapsed time = {}".format(end - start))


if __name__ == "__main__":
    main()
