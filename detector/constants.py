import tensorflow as tf

# for fpn only
MIN_LEVEL = 3
# the minimal feature stride that will be used is `2**MIN_LEVEL`

DIVISOR = 128

DATA_FORMAT = 'channels_first'

# a small value
EPSILON = 1e-8

# this is used when we are doing box encoding/decoding
SCALE_FACTORS = [10.0, 10.0, 5.0, 5.0]

# input pipeline settings
SHUFFLE_BUFFER_SIZE = 5000
NUM_PARALLEL_CALLS = 8

# images are resized before feeding them to the network
RESIZE_METHOD = tf.image.ResizeMethod.NEAREST_NEIGHBOR

# thresholds for IoU when creating training targets
POSITIVES_THRESHOLD = 0.5
NEGATIVES_THRESHOLD = 0.4

# this is used in tf.map_fn when creating training targets or doing NMS
PARALLEL_ITERATIONS = 8

# this can be important
BATCH_NORM_MOMENTUM = 0.95

# it is important to set this value
BATCH_NORM_EPSILON = 1e-3
