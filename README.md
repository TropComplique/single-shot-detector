# SSD with FPN in tensorflow

| Backbone      | AP | AP@0.5 | inference time |
| ----------- | ----------- | --- | --- | 
| Mobilenet v1      |    31.7    | 49.8 |36 ms |
| Shufflenet v2   | 29.3        | 46.7 | 31 ms | 

896, 640

## Requirements
* tensorflow 1.12
* tqdm, Pillow, numpy, matplotlib

## Credit
This implementation is based on [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
