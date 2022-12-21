# Presentation

The Aerial Image Recognition Challenge consists in sorting different landscape images ranging from beaches, lakes, forests, meadows and more.

# Data

Data comes from part of the data set (NWPU-RESISC45) originally used in the paper Remote Sensing Image Scene Classification. This data set contains 45 categories while we only kept 13 out of them as a first preprocessing.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/data_example.png?raw=true)" width="1000" height="800"/></p>

# Development

This solution to the Aerial Image Recognition Challenge was developed using [Jupyter](https://jupyter.org/) over the course of 2 months. The group members were KHATER Yara, [POBELLE Marion](https://github.com/marionpobelle) and [SERRE Gaëtan](https://github.com/gaetanserre).

We first used an instance of the REsNet50 model that is already trained on the [ImageNet](https://www.image-net.org/) dataset. We obtained stable results for our loss and accuracy.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/lossac_pretrained.png?raw=true)" width="1000" height="400"/></p>

As the pretrained ResNet50 model was really performant, we decided to recreate its architecture but without any pretrained weights. Once again we obtained great results on loss and accuracy.

<p align="center"><img src="https://github.com/marionpobelle/Aerial/blob/main/Aerial/img/lossac_untrained.png?raw=true)" width="1000" height="400"/></p>

