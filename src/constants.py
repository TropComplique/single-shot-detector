import tensorflow as tf

# a small value
EPSILON = 1e-8
# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

# Here are input pipeline settings.
# you need to tweak these numbers for your system,
# it can accelerate training
SHUFFLE_BUFFER_SIZE = 300
PREFETCH_BUFFER_SIZE = 100
NUM_THREADS = 8
# read here about buffer sizes:
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle

# images are resized before feeding to the network
RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR
# threshold for IoU when creating training targets
MATCHING_THRESHOLD = 0.5

# for tf.map_fn when creating training targets or doing nms
PARALLEL_ITERATIONS = 8

# this is important
BATCH_NORM_MOMENTUM = 0.9
