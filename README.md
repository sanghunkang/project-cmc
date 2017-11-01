## Project CMC
[한국어 README](./README.ko.md)

#### Dependencies
* Ubuntu 16.04 LTS
* Python 3.5.2

* CUDA 8.0.61
* CuDNN 8.0

* matplotlib 2.1.0
* numpy 1.13.3
* openCV 3.1.2 (?)
* pillow 3.1.2
* tensorFlow 1.3.0

Deployments in Windows and OSX are not supported yet.

## Usage
It follows the general convention of CLI programmes. Should you have questions of the detailed usage of each script, `python3 [scriptname].py -h` will print a kind help message in most cases.

## Data Augmentation
*yet to be implemented*

## Data Generataion
We rely on pickles of numpy.ndarray for now. 

* Example
```
python3 datagen.py -s /dev-root/ -d /dev-root/ -r 224 -f class_123 -e 1 -n 2
```

If you want a random separation of train and validation sets, you 
```
python3 datagen.py -d ... 
# for generation of a 
python3 splitdata.py -r 0.8 -f /dev-root/ -d /dev-root/ 
```

## Training

* Example
```
python3 train.py --dir_data_train=/dev-root/dev-data/diretory-for-train-data /
--dir_data_eval=/dev-root/dev-data/directory-for-validation-data /
--batch_size=128 /
--ckpt_name=checkpoint_my-model /
--num_steps=1000 /
--first_gpu_id=0 /
--num_gpu=2 /
--learning_rate=0.0001
```
This command will trigger the script `train.py` to use files at `/dev-root/dev-data/diretory-for-train-data` as training files and `/dev-root/dev-data/directory-for-validation-data` as validation files. The batch size at each iteration will be 128, and the ratio among classes will be balanced - which is not controllable by system arguments. The result will be stored in the checkpoint file named `checkpoint_my-model.ckpt`. The network will undergo 1000 iteration of weight updates with the learning rate of 0.0001. For computation, the programme will use 2 GPUs from the allocated ID of 0 by ascending order. 

## Inference

* Example
```
python3 inference.py --dir_data_inference=/  --ckpt_name=
```

#### I don't like this. I'm modifying this!