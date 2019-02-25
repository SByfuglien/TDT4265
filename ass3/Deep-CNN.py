from CNN import CNN
import keras
from keras.layers import Dense, Flatten, Dropout, Activation, BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import SGD

import time


class DeepCNN(CNN):

    def __init__(self):
        super().__init__()

    def model_construction(self, learning_rate):
        self.model.add(Conv2D(32, (3, 3), input_shape=self.X_train.shape[1:], padding='same'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.1))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(512))
        self.model.add(Activation('elu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.2))

        self.model.add(Dense(10))
        self.model.add(Activation('softmax'))

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
    cnn = DeepCNN()
    cnn.setup()
    cnn.model_construction(learning_rate)
    cnn.model_train(num_epochs, batch_size, data_aug=False)
    cnn.model_analysis()
    cnn.model.save('models/Deep-CNN')

    end = time.time()
    print("elapsed time = {:04.2f}".format(end - start))


if __name__ == "__main__":
    main()
