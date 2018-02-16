# DeepLearningModels
 Deep neural networks implemented in keras to finetune in your dataset
 
 We implemented the main deep models and extensions. In most of them, we just modify the source code to finetune for your dataset with pretrained models and it is possible to save the weights after it without lose the last layer weights when load the pre trained file later.
 
 You just need to implement a code to load your dataset and modify some lines of code such as, number of classes, number of batches, number of epochs.
 
 Requirements:
 
 Keras 2.0
 Theano
 
 
 ## AlexNet
 
 I modified the implementation of https://github.com/heuritech/convnets-keras to finetune in your dataset.

 
## Caffenet

I modified the implementation of https://github.com/yjn870/keras-caffenet to finetune in your dataset.

## ZFNet

In this implementation, I just modify the AlexNet implementation and load the Imagenet weights from Alexnet to finetune the Cifar 10. Note that I just Cifar 10 as an example and you can modificy to apply in your own dataset. 

## TO DO

 - Provide an example to show how to use the models in your dataset instead of use MNIST or CIFAR. I can not post the dataset bacause it is not available to be used yet.
 
 - Modify the models to be used in different image resolutions. 
