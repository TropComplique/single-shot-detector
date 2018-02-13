import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.constants import DATA_FORMAT


def mobilenet_v1_base(images, is_training, min_depth=8, depth_multiplier=1.0):
    """
    Arguments:
        images: a float tensor with shape [batch_size, 3, height, width],
            a batch of RGB images with pixels values in the range [0, 1].
        is_training: a boolean.
        min_depth: an integer, the minimal number of filters in a layer.
        depth_multiplier: a float number, possible values are [0.25, 0.5, 0.75, 1.0],
            multiplier for the number of filters in a layer.
    Returns:
        x: a float tensor with shape [batch_size, final_channels, final_height, final_width].
        features: a dict, layer name -> a float tensor.
    """

    def depth(x):
        """Get the number of filters for a layer."""
        return max(int(x * depth_multiplier), min_depth)

    def preprocess(images):
        """Transform images before feeding them to the network."""
        return (2.0*images) - 1.0

    def batch_norm(x):
        x = slim.batch_norm(
            x, center=True, scale=True,
            decay=0.9997, epsilon=0.001, is_training=is_training,
            fused=True, data_format=DATA_FORMAT
        )
        return x

    with tf.variable_scope('MobilenetV1', values=[images]):
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': DATA_FORMAT
        }
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], **params):
            features = {}
            x = preprocess(images)

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
                x = slim.separable_conv2d(x, None, (3, 3), depth_multiplier=1, stride=stride, scope=layer_name)
                features[layer_name] = x

                layer_name = 'Conv2d_%d_pointwise' % i
                x = slim.conv2d(x, depth(num_filters), (1, 1), stride=1, scope=layer_name)
                features[layer_name] = x

    return x, features
