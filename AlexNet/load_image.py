import PIL
import os
from PIL import Image
import glob
import cv2
import numpy as np
from scipy.misc import imsave
import scipy.misc


def Load_Image(src, num_channels, img_rows, n_class):

  X = np.zeros(img_rows*img_rows*num_channels)
  X.resize(1,img_rows,img_rows,num_channels)
  Y = np.zeros(n_class)
  Y.resize(1,n_class)

  ##=================abre as imagens para obter o label e ID======================
  basewidth=img_rows

  base=os.path.basename(src)
  f = os.path.splitext(base)[0]
  Label, ID = f.split("_")
    

    ##=================Abre as imagens ==============================================
  img=Image.open(src)
    ##================resize=====================================
  wpercent = (basewidth / float(img.size[0]))
  hsize = int((float(img.size[1]) * float(wpercent)))
  img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
  im = np.array(img)
    ##===========================================================

  X[0,:,:,:] = np.copy(im[:,:,:])

  Y[0,int(Label)-1] = 1

    ##=================Preprocessing===========================
  # Switch RGB to BGR order 
  X = X[:,:,:,::-1].copy()
  X1 = np.arange(basewidth*basewidth*3)
  X1.resize(1,basewidth,basewidth,3)

  # Subtract ImageNet mean pixel 
  X1[:, :, :, 0] = X[:, :, :, 0] -103.939
  X1[:, :, :, 1] = X[:, :, :, 1] -116.779
  X1[:, :, :, 2] = X[:, :, :, 2] -123.68


  ##===================================================
  
  ## Split the dataset into Train and test set

  return X1, Y 


def Load_Image_theano(src, num_channels, img_rows, n_class):
 
  X = np.zeros(img_rows*img_rows*num_channels)
  X.resize(1,img_rows,img_rows,num_channels)
  Y = np.zeros(n_class)
  Y.resize(1,n_class)

  ##=================abre as imagens para obter o label e ID======================
  basewidth=img_rows
  
  base=os.path.basename(src)
  f = os.path.splitext(base)[0]
  Label, ID = f.split("_")
    

    ##=================Abre as imagens ==============================================
  img=Image.open(src)
    ##================resize=====================================
  wpercent = (basewidth / float(img.size[0]))
  hsize = int((float(img.size[1]) * float(wpercent)))
  img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
  im = np.array(img)
    ##===========================================================

  X[0,:,:,:] = np.copy(im[:,:,:])

  Y[0,int(Label)-1] = 1

    ##=================Preprocessing===========================
  # Switch RGB to BGR order 
  X = X[:,:,:,::-1].copy()
  X1 = np.arange(basewidth*basewidth*3)
  X1.resize(1,3, basewidth,basewidth)

  # Subtract ImageNet mean pixel 
  X1[:, 0, :, :] = X[:, :, :, 0] -103.939
  X1[:, 1, :, :] = X[:, :, :, 1] -116.779
  X1[:, 2, :, :] = X[:, :, :, 2] -123.68


  ##===================================================
  
  ## Split the dataset into Train and test set

  return X1, Y

