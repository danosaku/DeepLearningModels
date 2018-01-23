# Keras CaffeNet

Implementation of CaffeNet in Keras 2.0 & Theano 0.9.0.

## Requirements

You need to install the following python library.

- Keras (2.0.2)
- Theano (0.9.0)
- Pillow (4.0.0)
- h5py (2.7.0)

## Download pre-trained weights

This weights file is converted from [bvlc_reference_caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet) to Keras format.

Original model was trained by Jeff Donahue @jeffdonahue.

[Download](https://drive.google.com/file/d/0B3H1zuduGkKXUEl2cERFczByVTQ/view?usp=sharing)

## Example

    python classify.py

    [('n02129604', 'tiger', 0.9994967), ('n02123159', 'tiger_cat', 0.00050332409), ('n02111500', 'Great_Pyrenees', 7.3053863e-10), ('n02120079', 'Arctic_fox', 2.0929633e-11), ('n02111889', 'Samoyed', 1.1360767e-11)]


or if you want to finetune the model to run in your own dataset use

    python finetune.py




## License

[MIT license](http://opensource.org/licenses/MIT)
