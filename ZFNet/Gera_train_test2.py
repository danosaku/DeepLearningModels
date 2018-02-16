import os
import sys
import numpy as np
from PIL import Image
import PIL
import cv2


def n_samples(path):
  num = 0
  ref_arquivo = open(path,"r")
  for linha in ref_arquivo:
     num+=1
  ref_arquivo.close()
  return num


 
def  load_dataset(num_channels, training_set_path, testing_set_path, nclass, img_rows):
  basewidth=img_rows
  src="/home/osaku/Eggs/pre/orig/"
  #==================Gera training Set
  n_train = n_samples(training_set_path)
  n_test = n_samples(testing_set_path)
  X_train = np.arange(n_train*basewidth*basewidth*num_channels)
  X_train.resize(n_train,basewidth*basewidth*num_channels)
  Y_train = np.zeros(n_train*nclass)
  Y_train.resize(n_train,nclass)

  ref_arquivo = open(training_set_path,"r")
  i = 0
  for linha in ref_arquivo:
    base=os.path.basename(linha)
    f = os.path.splitext(base)[0]
    path = src+f+".png"
    Label, ID = f.split("_")
    
##=================Abre as imagens ===============

    img=Image.open(path)
    #image_list.append(im)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)


#====================================================
   
    im = np.array(img)
    img = im.reshape(1,basewidth*basewidth*3)
    X_train[i,:] = np.copy(img[0,:])
    #print (X[i,1], img[0,1])
    #Y_train[i,int(Label)-1] = 1
    if (int(Label)>8):
      Y_train[i,8] = 1
    else:
      Y_train[i,int(Label)-1] = 1
    #print (Y[i,0], Y[i,1])
    i+=1
  X_train = X_train.reshape(-1,basewidth,basewidth,3)

  # Switch RGB to BGR order 
  X_train = X_train[:,:,:,::-1].copy()
  X1_train = np.arange(n_train*basewidth*basewidth*3)
  X1_train.resize(n_train,basewidth,basewidth,3)

  # Subtract ImageNet mean pixel 
  X1_train[:, :, :, 0] = X_train[:, :, :, 0] -103.939
  X1_train[:, :, :, 1] = X_train[:, :, :, 1] -116.779
  X1_train[:, :, :, 2] = X_train[:, :, :, 2] -123.68
  del X_train
  #X1_train = X1_train.reshape(-1, 224*224*3)

  ref_arquivo.close()

##========================== Gera testing set========

  X_test = np.arange(n_test*basewidth*basewidth*num_channels)
  X_test.resize(n_test,basewidth*basewidth*num_channels)
  Y_test = np.zeros(n_test*nclass)
  Y_test.resize(n_test,nclass)


  ref_arquivo = open(testing_set_path,"r")
  i = 0
  for linha in ref_arquivo:
    base=os.path.basename(linha)
    f = os.path.splitext(base)[0]
    path = src+f+".png"
    Label, ID = f.split("_")
    
##=================Abre as imagens ===============

    img=Image.open(path)
    #image_list.append(im)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    im = np.array(img)
    img = im.reshape(1,basewidth*basewidth*num_channels)
    X_test[i,:] = np.copy(img[0,:])
    #print (X[i,1], img[0,1])
    #Y_test[i,int(Label)-1] = 1
    if (int(Label)>8):
      Y_test[i,8] = 1
    else:
      Y_test[i,int(Label)-1] = 1
    i+=1

  X_test = X_test.reshape(-1,basewidth,basewidth,3)
  # Switch RGB to BGR order 
  X_test = X_test[:,:,:,::-1].copy()

  X1_test = np.arange(n_test*basewidth*basewidth*3)
  X1_test.resize(n_test,basewidth,basewidth,3)

  # Subtract ImageNet mean pixel 
  X1_test[:, :, :, 0] = X_test[:, :, :, 0] -103.939
  X1_test[:, :, :, 1] = X_test[:, :, :, 1] -116.779
  X1_test[:, :, :, 2] = X_test[:, :, :, 2] -123.68

  #X1_test = X1_test.reshape(-1, basewidth*basewidth*3)
  del X_test
  ref_arquivo.close()

  return X1_train, Y_train, X1_test, Y_test




def  load_dataset_theano(num_channels, training_set_path, testing_set_path, nclass, img_rows):
  basewidth=img_rows
  src="/home/osaku/Eggs/pre/orig/"
  #==================Gera training Set
  n_train = n_samples(training_set_path)
  n_test = n_samples(testing_set_path)
  X_train = np.arange(n_train*basewidth*basewidth*num_channels)
  X_train.resize(n_train,basewidth*basewidth*num_channels)
  Y_train = np.zeros(n_train*nclass)
  Y_train.resize(n_train,nclass)

  ref_arquivo = open(training_set_path,"r")
  i = 0
  for linha in ref_arquivo:
    base=os.path.basename(linha)
    f = os.path.splitext(base)[0]
    path = src+f+".png"
    Label, ID = f.split("_")
    
##=================Abre as imagens ===============

    img=Image.open(path)
    #image_list.append(im)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)


#====================================================
   
    im = np.array(img)
    img = im.reshape(1,basewidth*basewidth*3)
    X_train[i,:] = np.copy(img[0,:])
    #print (X[i,1], img[0,1])
    #Y_train[i,int(Label)-1] = 1
    if (int(Label)>8):
      Y_train[i,8] = 1
    else:
      Y_train[i,int(Label)-1] = 1
    #print (Y[i,0], Y[i,1])
    i+=1
  X_train = X_train.reshape(-1,basewidth,basewidth,3)

  # Switch RGB to BGR order 
  X_train = X_train[:,:,:,::-1].copy()
  X1_train = np.arange(n_train*basewidth*basewidth*3)
  X1_train.resize(n_train,3, basewidth,basewidth)

  # Subtract ImageNet mean pixel 
  X1_train[:,0, :, :] = X_train[:, :, :, 0] -103.939
  X1_train[:,1, :, :] = X_train[:, :, :, 1] -116.779
  X1_train[:,2, :, :] = X_train[:, :, :, 2] -123.68
  del X_train
  #X1_train = X1_train.reshape(-1, 224*224*3)

  ref_arquivo.close()

##========================== Gera testing set========

  X_test = np.arange(n_test*basewidth*basewidth*num_channels)
  X_test.resize(n_test,basewidth*basewidth*num_channels)
  Y_test = np.zeros(n_test*nclass)
  Y_test.resize(n_test,nclass)


  ref_arquivo = open(testing_set_path,"r")
  i = 0
  for linha in ref_arquivo:
    base=os.path.basename(linha)
    f = os.path.splitext(base)[0]
    path = src+f+".png"
    Label, ID = f.split("_")
    
##=================Abre as imagens ===============

    img=Image.open(path)
    #image_list.append(im)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)

    im = np.array(img)
    img = im.reshape(1,basewidth*basewidth*num_channels)
    X_test[i,:] = np.copy(img[0,:])
    #print (X[i,1], img[0,1])
    #Y_test[i,int(Label)-1] = 1
    if (int(Label)>8):
      Y_test[i,8] = 1
    else:
      Y_test[i,int(Label)-1] = 1
    i+=1

  X_test = X_test.reshape(-1,basewidth,basewidth,3)
  # Switch RGB to BGR order 
  X_test = X_test[:,:,:,::-1].copy()

  X1_test = np.arange(n_test*basewidth*basewidth*3)
  X1_test.resize(n_test, 3, basewidth,basewidth)

  # Subtract ImageNet mean pixel 
  X1_test[:, 0, :, :] = X_test[:, :, :, 0] -103.939
  X1_test[:, 1, :, :] = X_test[:, :, :, 1] -116.779
  X1_test[:, 2, :, :] = X_test[:, :, :, 2] -123.68

  #X1_test = X1_test.reshape(-1, basewidth*basewidth*3)
  del X_test
  ref_arquivo.close()

  return X1_train, Y_train, X1_test, Y_test

