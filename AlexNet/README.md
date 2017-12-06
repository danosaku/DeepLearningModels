# My experiments with AlexNet, using Keras and Theano
A blog post accompanying this project can be found [here](https://rahulduggal2608.wordpress.com/2017/04/02/alexnet-in-keras/).

## Contents
1. [Motivation](#motivation)
2. [Requirements](#requirements)
3. [Experiments](#experiments)
4. [Results](#results)
5. [TO-DO](#to-do)
8. [License](#license)

## Motivation
When I first started exploring deep learning (DL) in July 2016, many of the papers I read established their baseline performance using the standard AlexNet model. In part, this could be attributed to the several code examples readily available across all major Deep Learning libraries. Despite its significance, I could not find readily available code examples for training AlexNet in the Keras framework. Through this project, I am sharing my experience of training AlexNet in three very useful scenarios :-

1. **Training AlexNet end-to-end** - Also known as training from scratch
2. **Fine-Tuning the pre-trained AlexNet** - extendable to transfer learning
3. **Using AlexNet as a feature extractor** - useful for training a classifier such as SVM on top of "Deep" CNN features.

2. [This](https://github.com/heuritech/convnets-keras) project by Heuritech, which has implemented the AlexNet architecture.

## Requirements
This project is compatible with **Python 2.7-3.5**
Make sure you have the following libraries installed.
1. [Keras](https://keras.io) - A high level neural network library written in python. To install, follow the instructions available [here](https://keras.io/#installation).
2. [Theano](http://deeplearning.net/software/theano/introduction.html) - A python library to efficiently evaluate/optimize mathematical expressions. To install, follow the instructions available [here](http://deeplearning.net/software/theano/install.html).
3. [Anaconda](https://docs.continuum.io/) - A package of python libraries which includes several that are absolutely useful for Machine Learning/Data Science. To install, follow the instructions available [here](https://docs.continuum.io/anaconda/install). 

**Note :** If you have a GPU in your machine, you might want to configure Keras and Theano to utilize its resources. For myself, running the code on a K20 GPU resulted in a 10-12x speedup.



- Download the pre-trained weights for alexnet from [here](http://files.heuritech.com/weights/alexnet_weights.h5) and place them in ```convnets-keras/weights/```.
- Once the dataset and weights are in order, navigate to the project root directory, and run the command ```jupyter notebook``` on your shell. This will open a new tab in your browser. Navigate to ```Code/``` and open the file ```AlexNet_Experiments.ipynb```.
- Now you can execute each code cell using ```Shift+Enter``` to generate its output.

## TO-DO
1. The mean subtraction layer (look inside Code/alexnet_base.py) currently uses a theano function - set_subtensor. This introduces a dependancy to install Theano. I would ideally like to use a keras wrapper function which works for both Theano and Tensorflow backends. I'm not sure if such a wrapper exists though. Any suggestions for the corresponding Tensorflow function, so that I could write the Keras wrapper myself?
2. Use this code to demonstrate performance on a dataset that is significantly different from ImageNet. Maybe a medical imaging dataset?

## License
This code is released under the MIT License (refer to the LICENSE file for details).



