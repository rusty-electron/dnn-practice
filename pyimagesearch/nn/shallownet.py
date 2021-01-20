from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras import backend as k

class ShallowNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        if k.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model


