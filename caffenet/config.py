
##base de eggs
import Gera_train_test2
##base de Protozoan
import Generate_dataset_Protozoan
##base de Larvae
import Generate_dataset_Larvae
from datetime import datetime

def Load():

  save_weights=True
  batch_size = 16 
  nb_epoch = 10

  num_classes = 10 #for cifar10

  return batch_size, nb_epoch, num_classes, save_weights



def Load_dataset(img_rows, num_classes, channel, backend):
    #base de eggs
    if backend=="Tensorflow":
      if base=="Eggs":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Gera_train_test2.load_dataset_imagenet_preprocessing(channel, train_file, test_file, num_classes, img_rows)   
        else:
          X_train, Y_train, X_valid, Y_valid = Gera_train_test2.load_dataset_without_pretrain(channel,train_file, test_file, num_classes, img_rows)

      #base de Protozoan
      if base=="Protozoan":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Generate_dataset_Protozoan.load_dataset_tensorflow(channel, train_size, num_classes, img_rows)

      #base de Larvae
      if base=="Larvae":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Generate_dataset_Larvae.load_dataset_tensorflow(channel, train_size, num_classes, img_rows)
    else:
      if base=="Eggs":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Gera_train_test2.load_dataset_theano_imagenet_preprocessing(channel, train_file, test_file, num_classes, img_rows)   
        else:
          X_train, Y_train, X_valid, Y_valid = Gera_train_test2.load_dataset_theano_without_pretrain(channel,train_file, test_file, num_classes, img_rows)

      #base de Protozoan
      if base=="Protozoan":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Generate_dataset_Protozoan.load_dataset_theano(channel, train_size, num_classes, img_rows)

      #base de Larvae
      if base=="Larvae":
        if pretrain:
          X_train, Y_train, X_valid, Y_valid = Generate_dataset_Larvae.load_dataset_theano(channel, train_size, num_classes, img_rows)


    return X_train, Y_train, X_valid, Y_valid

def load_target_names():
  if base=="Eggs":
    target_names = Gera_train_test2.load_target_names()
  else:
    #base de Protozoan
    if base=="Protozoan":
      target_names = Generate_dataset_Protozoan.load_target_names()
    else:
      ##base de Larvae
      if base=="Larvae":
         target_names = Generate_dataset_Larvae.load_target_names()
   
  return target_names


def Get_filename_weights_to_save(model, mb, ne):

  data = datetime.now()


  if pretrain:
    filename = base+"_"+model+"_mb_"+str(mb)+"_ne_"+str(ne)+"_pretrained_weights("+str(data.day)+"-"+str(data.month)+"-"+str(data.year)+"-"+str(data.hour)+"-"+str(data.minute)+").h5"
  else:
    filename = base+"_"+model+"_mb_"+str(mb)+"_ne_"+str(ne)+"_without_pretrain_weights("+str(data.day)+"-"+str(data.month)+"-"+str(data.year)+"-"+str(data.hour)+"-"+str(data.minute)+").h5"  
  return filename
