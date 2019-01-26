# SSD with FPN (RetinaNet) in `tensorflow`

| Backbone | AP | AP@0.5 | inference time |
| --- | --- | --- | --- |
| Mobilenet v1 | 31.7 | 49.8 |36 ms |
| Shufflenet v2 | 29.3 | 46.7 | 31 ms |

## Notes
1. Inference time was measured on images of size 896x640 and using NVIDIA GTX 1080 Ti.
2. Average precision was computed on COCO val 2017 dataset.
3. During the evaluation I resize smallest dimension to be equal to 640 (while keeping the aspect ratio).
4. You can get the pretrained models from [here](https://drive.google.com/open?id=1sq57Fn3Ho1T4JLhWGOc13XpM6VZDuHD4).  
You can test them using `inference/just_try_detector.ipynb`.

## How to use this
1. Prepare dataset. See `data/README.md`.
2. Edit a json configuration file and then just run `python train.py`.
3. Then use `create_pb.py` to get a frozen inference graph.
4. Use notebooks in `inference` folder to test the trained detector.

## Requirements
* tensorflow 1.12
* tqdm, Pillow, numpy, matplotlib

## Credit
This implementation is based on [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).
