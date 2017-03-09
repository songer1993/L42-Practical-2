# import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Model
from keras.layers import Input, Dense, Flatten

def get_baseline_model(rows=28, cols=28, hidden_size=128, nb_classes=10, activation='sigmoid'):
    input_shape = (rows, cols, 1)

    inp = Input(shape=input_shape)
    flat = Flatten()(inp)
    hidden_1 = Dense(hidden_size, activation=activation)(flat)
    hidden_2 = Dense(hidden_size, activation=activation)(hidden_1)
    out = Dense(nb_classes, activation='softmax')(hidden_2)
    
    model = Model(input=inp, output=out)

    print(model.summary())

    return model


def get_cnn_model(width=28, height=28, depth=1, hidden_size=128, nb_classes=10, activation='sigmoid'):
    # Input layer
    input_shape = (height, width, 1)
    inp = Input(shape=input_shape)

    # first set of CONV => RELU => POOL
    conv1 = Convolution2D(20, 5, 5, border_mode="same")(inp)
    relu1 = Activation("relu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(relu1)

    # second set of CONV => RELU => POOL
    conv2 = Convolution2D(50, 5, 5, border_mode="same")(pool1)
    relu2 = Activation("relu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(relu2)

    # set of FC => RELU layers
    flat3 = Flatten()(pool2)
    fc3 = Dense(500)(flat3)
    relu3 = Activation("relu")(fc3)

    # softmax classifier
    fc4 = Dense(nb_classes)(relu3)
    out = Activation("softmax")(fc4)

    model = Model(input=inp, output=out)

    print(model.summary())
    # return the constructed network architecture
    return model



if __name__ == '__main__':

    model = get_cnn_model()
