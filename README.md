# Siamese Network for classification, segmentation and matching of multiview mammography 

Projet 3A - IMT Atlantique 

![img](https://github.com/alixlam/siamese_net_multi_task/blob/main/images/Screenshot%202021-03-25%20at%2016.33.56.png)


## Table of contents 

* [Files](#file)
* [Run code](#run)
*

## Files <a name="file"></a>

|files|Description|
|-----|---|
|[config.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/config.py)|training and dataloader parameters|
|[losses.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/losses.py)|Different losses tested for different tasks|
|[lightningModel.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/lightningModel.py)|lightning model|
|[models.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/models.py)|Encoder and Decoder models|
|[train.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/train.py)|example of code for training|
|[utils.py](https://github.com/alixlam/siamese_net_multi_task/blob/main/utils.py)|Useful functions for displays|
|[mammoDataPair.py](https://github.com/alixlam/siamese_net_multi_task/tree/main/data/mammoDataPair.py)|loader for data pair (CC and MLO view)|
|[mammoDataSingle.py](https://github.com/alixlam/siamese_net_multi_task/tree/main/data/mammoDataSingle.py)|loading for single data (CC or MLO view)|
|[datamodule.py](https://github.com/alixlam/siamese_net_multi_task/tree/main/data/datamodule.py)|Wrapper around mammoDataPair and single for lightning module|

## Run code <a name="run"></a>

Examples of python notebook can be found [here](https://github.com/alixlam/siamese_net_multi_task/tree/main/Colab_notebooks).

