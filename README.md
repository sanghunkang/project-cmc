## Project CMC
[한국어 README](./README.ko.md)

#### Dependencies
Ubuntu 16.04 LTS

Python 3.5.2
TensorFlow 1.3.0
numpy 1.13...
PIL

CUDA
CUDNN
possibly some others?
Deployments in Windows and OSX are not supported yet.

## Usage
In general, `python3 [scriptname].py -h` will print a kind report of the usage

#### Data Augmentation
*yet to be implemented*

#### Data Generataion
We rely on pickles of numpy.ndarray for now. 

* Example
```
python3 datagen.py -d ...
```

#### Train

* Example
```
python3 train.py --dir_data_train==....
```

#### Inference

* Example
```
python3 inference.py --dir_data_inference==....
```
