import os
import sys
os.environ["KERAS_BACKEND"] = "theano"
import keras.backend as k 
k.set_image_dim_ordering('th')
sys.path.insert(0, 'preprocess/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'layers/')
from sklearn.metrics import log_loss, classification_report
import Generate_train_test2
import config
from caffenet import CaffeNet
from keras.applications.imagenet_utils import decode_predictions
import numpy as np
from load_cifar10 import load_cifar10_data

if __name__ == '__main__':
    finetune = True
    img_rows, img_cols = 227, 227 # Resolution of inputs
    channel = 3
    batch_size, nb_epoch, num_classes, save_weights = config.Load()
    # Load Cifar10 data. Please implement your own load_data() module for your own dataset
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
    
    model = CaffeNet(weights='caffenet_weights_th.h5', classes=num_classes)

    # Start Fine-tuning
    if finetune:
       model.fit(X_train, Y_train,
                batch_size=batch_size,
                nb_epoch=nb_epoch,
                shuffle=True,
                verbose=1,
                validation_data=(X_valid, Y_valid),
                )

    #model.save('Eggs_weights/caffenet_weights_th.h5')
    if save_weights:
      filename_weights = config.Get_filename_weights_to_save('Caffenet', batch_size, nb_epoch)
      model.save(filename_weights)
    # Make predictions
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    #del model
    # Cross-entropy loss score
    score = log_loss(Y_valid, predictions_valid)
    target_names = config.load_target_names()
    #target_names = Generate_train_test2.load_target_names()
    #print Y_valid, predictions_valid
    Y_test = np.argmax(Y_valid, axis=-1) # Convert one-hot to index
    Y_pred = np.argmax(predictions_valid, axis=-1)
    #y_pred = model.predict_classes(X_valid)
    print(classification_report(Y_test, Y_pred, target_names=target_names))


    del model
