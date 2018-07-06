
from datetime import datetime

def Load():

  save_weights=True
  batch_size = 16 
  nb_epoch = 10

  num_classes = 10 #for cifar10

  return batch_size, nb_epoch, num_classes, save_weights


def Get_filename_weights_to_save(model, mb, ne):

  data = datetime.now()


  if pretrain:
    filename = model+"_mb_"+str(mb)+"_ne_"+str(ne)+"_pretrained_weights("+str(data.day)+"-"+str(data.month)+"-"+str(data.year)+"-"+str(data.hour)+"-"+str(data.minute)+").h5"
  else:
    filename = model+"_mb_"+str(mb)+"_ne_"+str(ne)+"_without_pretrain_weights("+str(data.day)+"-"+str(data.month)+"-"+str(data.year)+"-"+str(data.hour)+"-"+str(data.minute)+").h5"  
  return filename


def load_target_names():
    return ['Class 1','Class 2','Class 3','Class 4','Class 5','Class 6','Class 7','Class 8','Class 9','Class 10']
