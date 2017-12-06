import os
import sys
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend as k 
k.set_image_dim_ordering('th')
import Generate_train_test2
from sklearn.metrics import log_loss, classification_report
from load_cifar10 import load_cifar10_data
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D
#from imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids


def convnet(network, weights_path=None,
            trainable=None, num_classes=1000):
    """
    Returns a keras model for a CNN.

    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of img_rows and img_cols for each CNN :
    * For AlexNet, the data are of shape (227,227), and the colors in the RGB order (default)
    * For VGG16 and VGG19, the data are of shape (224,224), and the colors in the BGR order

    It can also be used to look at the hidden layers of the model.

    It can be used that way :

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
    if network == 'vgg_16':
        convnet_init = VGG_16
    elif network == 'vgg_19':
        convnet_init = VGG_19
    elif network == 'alexnet':
        convnet_init = AlexNet
    convnet = convnet_init(weights_path, heatmap=False, classes=num_classes)

    return convnet

def VGG_16(weights_path=None, heatmap=False, classes=1000):
    model = Sequential()
    if heatmap:
        model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
    else:
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if heatmap:
        model.add(Convolution2D(4096,7,7,activation="relu",name="dense_1"))
        model.add(Convolution2D(4096,1,1,activation="relu",name="dense_2"))
        model.add(Convolution2D(1000,1,1,name="dense_3"))
        model.add(Softmax4D(axis=1,name="softmax"))
    else:
        model.add(Flatten(name="flatten"))
        model.add(Dense(4096, activation='relu', name='dense_1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', name='dense_2'))
        model.add(Dropout(0.5))
        model.add(Dense(1000, name='dense_3'))
        model.add(Activation("softmax",name="softmax"))

    if weights_path:
        model.load_weights(weights_path)


    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Uncomment below to set the first 10 layers to non-trainable (weights will not be updated)
    #for layer in model.layers[:10]:
    #    layer.trainable = False

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



    return model

def VGG_19(weights_path=None,heatmap=False, classes=1000):
    model = Sequential()

    if heatmap:
        model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
    else:
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu', name='dense_1'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu', name='dense_2'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, name='dense_3'))
    model.add(Activation("softmax"))

    if weights_path:
        model.load_weights(weights_path)


   # Truncate and replace softmax layer for transfer learning
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes, activation='softmax'))

    # Learning rate is changed to 0.001
    sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])


    return model



def AlexNet(weights_path=None, heatmap=False, classes = 1000):
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
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
    dense_1 = Flatten(name="flatten")(dense_1)
    dense_1 = Dense(4096, activation='relu',name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu',name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)   
    dense_3 = Dense(classes, name='dense_3a')(dense_3)
    prediction = Activation("softmax",name="softmaxa")(dense_3)


    model = Model(input=inputs, output=prediction, name='alexnet')

    if weights_path:
        model.load_weights(weights_path, by_name=True)

    # Learning rate is changed to 0.0001
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
 

    img_rows, img_cols = 227, 227 # Resolution of inputs
    channel = 3
    num_classes = 10
    batch_size = 16 
    nb_epoch = 10

    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    

    # Here is just an example of how you can implement your code to load the dataset
    #X_train, Y_train, X_valid, Y_valid = Generate_train_test2.load_dataset_theano_imagenet_preprocessing(3, "eggs_train.txt", "eggs_test.txt", num_classes, img_rows)

    model = convnet('alexnet', weights_path="weights/alexnet_weights.h5", num_classes=num_classes)
    model.summary()

    # Start Fine-tuning
    model.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              shuffle=True,
              verbose=1,
              validation_data=(X_valid, Y_valid),
              )

    # If you want to save the model weights
    model.save('Eggs_weights/Alexnet_weights_th.h5')

    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

    #del model

    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    target_names = Generate_train_test2.load_target_names()
    #print Y_valid, predictions_valid
    Y_test = np.argmax(Y_valid, axis=-1) # Convert one-hot to index
    Y_pred = np.argmax(predictions_valid, axis=-1)
    #y_pred = model.predict_classes(X_valid)
    print(classification_report(Y_test, Y_pred, target_names=target_names))


