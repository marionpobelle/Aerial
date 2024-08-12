# Aerial
## Summary

The Aerial Image Recognition Challenge revolves around sorting different landscape images ranging from beaches, lakes, forests, meadows and more.

## Data

The data comes from parts of the data set (NWPU-RESISC45) originally used in the paper Remote Sensing Image Scene Classification. This data set contains 45 categories while we only kept 13 out of them as a first preprocessing.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/data_example.png?raw=true)" width="600" height="400"/></p>

## Development

This solution to the Aerial Image Recognition Challenge was developed using [Jupyter](https://jupyter.org/) over the course of 2 months. The group members were KHATER Yara, [POBELLE Marion](https://github.com/marionpobelle) and [SERRE Gaëtan](https://github.com/gaetanserre).

We first used an instance of the REsNet50 model that is already trained on the [ImageNet](https://www.image-net.org/) dataset. 
All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded into a range of [0, 1] (in order to have small weights for small data entries) and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225] (both mean and standard deviation with pixels).
So we made our images look the same way as the images ResNet50 was pre-trained on.
We obtained stable results for our loss and accuracy.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/lossac_pretrained.png?raw=true)" width="1000" height="300"/></p>

As the pretrained ResNet50 model was performing well, we decided to recreate its architecture but without any pretrained weights. Once again we obtained great results on loss and accuracy.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/lossac_untrained.png?raw=true)" width="1000" height="300"/></p>

