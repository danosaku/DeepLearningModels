import os
import sys
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend as k 
k.set_image_dim_ordering('th')
import Gera_train_test2
from sklearn.metrics import log_loss

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, concatenate, merge, Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave
from load_cifar10 import load_cifar10_data
from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D

def convnet(network, weights_path=None, 
            trainable=None, num_classes=1000):
    """
    Returns a keras model for a CNN.

    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of preprocess_image_batch for each CNN :
    * In AlexNet and ZFNet, the data are of shape (227,227), and the colors in the RGB order (default)

    It can also be used to look at the hidden layers of the model.


    >>> # Test pretrained model
    >>> model = convnet('vgg_16', 'weights/vgg16_weights.h5')
    >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    >>> out = model.predict(im)

    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'

    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained

    heatmap: bool
        Says wether the fully connected layers are transformed into Convolution2D layers,
        to produce a heatmap instead of a


    Returns
    ---------------
    model:
        The keras model for this convnet

    output_dict:
        Dict of feature layers, asked for in output_layers.
    """


    # Select the network
    if network == 'zfnet':
        convnet_init = ZFNet
    elif network == 'alexnet':
        convnet_init = AlexNet
    convnet = convnet_init(weights_path, classes=num_classes)

    return convnet
    

def ZFNet(weights_path=None, classes = 1000):

    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, (7, 7),strides=(2,2),kernel_initializer='random_uniform', bias_initializer='zeros', activation='relu',
                           name='conv_1a')(inputs)
    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = concatenate([
        Convolution2D(128,(5,5),strides=(2,2), activation="relu",name='conv_2'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,(3,3),activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = concatenate([
        Convolution2D(192,(3,3),activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = concatenate([
        Convolution2D(128,(3,3),activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(classes, kernel_initializer='random_uniform', bias_initializer='zeros', name='dense_3a')(dense_3)
    prediction = Activation("softmax",name="softmaxa")(dense_3)


    model = Model(input=inputs, output=prediction, name='zfnet')

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    return model


def AlexNet(weights_path=None, classes = 1000):

    inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, (11, 11),strides=(4,4),activation='relu',
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,(5,5),activation="relu",name='conv_2_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,(3,3),activation='relu',name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,(3,3),activation="relu",name='conv_4_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,(3,3),activation="relu",name='conv_5_'+str(i+1))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)
    emb1 = Embedding(256, classes, name='last_conv')(conv_5)

    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(classes,name='dense_3a')(dense_3)
    prediction = Activation("softmax",name="softmax1")(dense_3)


    model = Model(input=inputs, output=prediction, name='alexnet')

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    # Test pretrained model
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    return model



def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]

        img_list.append(img)

    try:
        img_batch = np.stack(img_list, axis=0)
    except:
        raise ValueError('when img_size and crop_size are None, images'
                ' in image_paths must have the same shapes.')

    if out is not None and hasattr(out, 'append'):
        out.append(img_batch)
    else:
        return img_batch





if __name__ == "__main__":
    ### 
    ## We find the synsets corresponding to dogs on ImageNet website

    img_rows, img_cols = 227, 227 # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    #X_train, Y_train, X_valid, Y_valid = Gera_train_test2.load_dataset_theano(3, "eggs_train.txt", "eggs_test.txt", num_classes, img_rows)

    model = convnet('zfnet', weights_path="weights/alexnet_weights.h5", num_classes=num_classes)

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    #model.save('ZFNet_th.h5')
    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    del model
    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)

    out = model.predict(im)
