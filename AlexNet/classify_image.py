# -*- coding: utf-8 -*-
import finetune
import load_image
import sys
import config
import time
import numpy as np

if __name__ == '__main__':
    start = time.time()

    src = sys.argv[1]
    img_rows, img_cols = 227, 227 # Resolution of inputs
    channel = 3
   
    batch_size, nb_epoch, num_classes, save_weights = config.Load()

    X_valid, Y_valid = load_image.Load_Image_theano(src, channel, img_rows, num_classes)

    # Load our model
    model = finetune.convnet('alexnet', weights_path="weights/alexnet_weights.h5", num_classes=num_classes)

    preds = model.predict(X_valid)

    Y_test = np.argmax(Y_valid, axis=-1) # Convert one-hot to index
    Y_pred = np.argmax(preds, axis=-1)
    print "Predicted:", Y_pred, "True Label:", Y_test

    end = time.time()
    total = end-start
    print "Execution time - ", end-start," sec"
