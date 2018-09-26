import tensorflow as tf
from detector.constants import BATCH_NORM_MOMENTUM, BATCH_NORM_EPSILON, DATA_FORMAT


def batch_norm_relu(x, is_training, use_relu=True, name=None):
    x = tf.layers.batch_normalization(
        inputs=x, axis=1 if DATA_FORMAT == 'channels_first' else 3,
        momentum=BATCH_NORM_MOMENTUM, epsilon=BATCH_NORM_EPSILON,
        center=True, scale=True, training=is_training,
        fused=True, name=name
    )
    return x if not use_relu else tf.nn.relu(x)


def conv2d_same(x, num_filters, kernel_size=3, stride=1, rate=1, name=None):
    if stride == 1:
        return tf.layers.conv2d(
            inputs=x, filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), dilation_rate=(rate, rate),
            padding='same', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=DATA_FORMAT, name=name
        )
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if DATA_FORMAT == 'channels_first':
            paddings = [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]]
        else:
            paddings = [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]]

        return tf.layers.conv2d(
            inputs=tf.pad(x, paddings), filters=num_filters,
            kernel_size=(kernel_size, kernel_size),
            strides=(stride, stride), dilation_rate=(rate, rate),
            padding='valid', use_bias=False,
            kernel_initializer=tf.variance_scaling_initializer(),
            data_format=DATA_FORMAT, name=name
        )
