from CNN import CNN
from keras.layers import Dense, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.losses import categorical_crossentropy



class table1(CNN):

    def __init__(self):
        super().__init__()

    def model_construction(self, learning_rate):
        self.model.add(Conv2D(32, (5, 5), input_shape=(32, 32, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(64, (5, 5), input_shape=(32, 32, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Conv2D(128, (5, 5), input_shape=(32, 32, 3), padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        self.model.add(Flatten())

        self.model.add(Dense(64))
        self.model.add(Activation('relu'))

        self.model.add(Dense(10, activation="softmax"))

        sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss=categorical_crossentropy,
                           optimizer=sgd,
                           metrics=['accuracy'])
        self.model.summary()


def main():
    learning_rate = 0.003
    num_epochs = 10
    batch_size = 64
    cnn = table1()
    cnn.setup()
    cnn.model_construction(learning_rate)
    cnn.model_train(num_epochs, batch_size, data_aug=True)
    cnn.model_analysis()
    cnn.model.save('models/Table1-CNN')


if __name__ == "__main__":
    main()