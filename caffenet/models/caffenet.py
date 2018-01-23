from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.optimizers import SGD
from lrn2d import LRN2D


def CaffeNet(weights=None, input_shape=(3, 227, 227), classes=1000):
    inputs = Input(shape=input_shape)
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv1')(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)
    x = LRN2D(name='norm1')(x)
    x = ZeroPadding2D((2, 2))(x)
    x = Conv2D(256, (5, 5), activation='relu', name='conv2')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool2')(x)
    x = LRN2D(name='norm2')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv3')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(384, (3, 3), activation='relu', name='conv4')(x)
    x = ZeroPadding2D((1, 1))(x)
    x = Conv2D(256, (3, 3), activation='relu', name='conv5')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool5')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc6')(x)
    x = Dropout(0.5, name='drop6')(x)
    x = Dense(4096, activation='relu', name='fc7')(x)
    x = Dropout(0.5, name='drop7')(x)
    xf = Dense(1000, name='fc8')(x)
    xf = Activation('softmax', name='loss')(xf)

    model = Model(inputs, xf, name='caffenet')

    model.load_weights(weights)

    x_newfc = Dense(classes, name='fc_t')(x)
    x_newfc = Activation('softmax', name='loss_t')(x_newfc)

    model = Model(input=inputs, output=x_newfc)

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
  
    model.summary()
    return model
