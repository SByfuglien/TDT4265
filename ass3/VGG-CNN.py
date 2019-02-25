from CNN import CNN
import keras
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.applications import vgg19

import time

class VGGCNN(CNN):


    def __init__(self):
        super().__init__()

    def model_construction(self, learning_rate):
        vgg = vgg19.VGG19(include_top=False,
                                   weights='imagenet',
                                   input_shape=self.X_train.shape[1:])

        for layer in vgg.layers[:-5]:
            layer.trainable = False

        self.model.add(vgg)

        self.model.add(Flatten())

        self.model.add(Dense(10, activation="softmax"))

        adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)
        self.model.compile(loss=keras.losses.categorical_crossentropy,
                           optimizer=adam,
                           metrics=['accuracy'])
        self.model.summary()


def main():
    start = time.time()
    learning_rate = 0.0005
    num_epochs = 10
    batch_size = 32
    cnn = VGGCNN()
    cnn.setup()
    cnn.model_construction(learning_rate)
    cnn.model_train(num_epochs, batch_size, data_aug=True)
    # cnn.model_analysis()
    cnn.model.save('models/ResNet-CNN')

    end = time.time()
    print("elapsed time = {}".format(end - start))


if __name__ == "__main__":
    main()
