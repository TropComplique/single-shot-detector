# SSD in tensorflow
**Warning:** this implementation works fine but isn't mature yet.

## Implementation details
* This implementation uses `tf.estimator` and `tf.data` frameworks.

## How to use it
* Convert your dataset into `.tfrecords` format using `create_tfrecords.py`.
* Edit `src/features.py` to set the feature extractor as you like.
* Run `train.py` for training a detector. Evaluation will happen periodically.
* Run `tensorboard` to observe the training process.  
Average precision on the validation set will be shown there.
* Run `save.py` to export the trained model for inference.

## Requirements
* Python 3.6
* tensorflow 1.4
* tqdm, Pillow, numpy, matplotlib

## Credit
This implementation is based on [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
