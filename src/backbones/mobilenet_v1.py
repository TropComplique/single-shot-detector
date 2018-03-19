import tensorflow as tf
import tensorflow.contrib.slim as slim

from src.constants import BATCH_NORM_MOMENTUM
from src.backbones.depthwise_conv import depthwise_conv


def mobilenet_v1_base(images, is_training, depth_multiplier=1.0, min_depth=8):
    """
    Arguments:
        images: a float tensor with shape [batch_size, 3, height, width],
            a batch of RGB images with pixel values in the range [0, 1].
        is_training: a boolean.
        depth_multiplier: a float number, multiplier for the number of filters in a layer.
        min_depth: an integer, the minimal number of filters in a layer.
    Returns:
        x: a float tensor with shape [batch_size, final_channels, final_height, final_width].
        features: a dict, layer name -> a float tensor.
    """

    def depth(x):
        """Reduce the number of filters in a layer."""
        return max(int(x * depth_multiplier), min_depth)

    def preprocess(images):
        """Transform images before feeding them to the network."""
        return (2.0*images) - 1.0

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
            training=is_training, fused=True,
            name='BatchNorm'
        )
        return x

    with tf.name_scope('standardize_input'):
        x = preprocess(images)

    with tf.variable_scope('MobilenetV1'):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }
        with slim.arg_scope([slim.conv2d, depthwise_conv], **params):
            features = {}

            layer_name = 'Conv2d_0'
            x = slim.conv2d(x, depth(32), (3, 3), stride=2, scope=layer_name)
            features[layer_name] = x

            strides_and_filters = [
                (1, 64),
                (2, 128), (1, 128),
                (2, 256), (1, 256),
                (2, 512), (1, 512), (1, 512), (1, 512), (1, 512), (1, 512),
                (2, 1024), (1, 1024)
            ]
            for i, (stride, num_filters) in enumerate(strides_and_filters, 1):

                layer_name = 'Conv2d_%d_depthwise' % i
                x = depthwise_conv(x, stride=stride, scope=layer_name)
                features[layer_name] = x

                layer_name = 'Conv2d_%d_pointwise' % i
                x = slim.conv2d(x, depth(num_filters), (1, 1), stride=1, scope=layer_name)
                features[layer_name] = x

    return x, features
