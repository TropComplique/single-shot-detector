import tensorflow as tf
import tensorflow.contrib.slim as slim
from src.constants import BATCH_NORM_MOMENTUM


class FeatureExtractor:
    def __init__(self, backbone, is_training):
        self.backbone = backbone
        self.is_training = is_training

    def __call__(self, images):
        """
        Arguments:
            images: a float tensor with shape [batch_size, 3, height, width],
                a batch of RGB images with pixels values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, channels_i, height_i, width_i].
        """

        x, feature_maps = self.backbone(images, self.is_training)
        image_features = [
            feature_maps['Conv2d_9_pointwise'],
            feature_maps['Conv2d_11_pointwise'],
            feature_maps['Conv2d_13_pointwise']
        ]

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=1, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=self.is_training, fused=True,
                name='BatchNorm'
            )
            return x

        filters = [128, 128, 128, 128]
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }
        with slim.arg_scope([slim.conv2d], **params):
            for i, num_filters in enumerate(filters, 14):
                x = slim.conv2d(x, num_filters // 2, (1, 1), stride=1, scope='Conv2d_%d_1x1' % i)
                x = slim.conv2d(x, num_filters, (3, 3), stride=2, scope='Conv2d_%d' % i)
                image_features.append(x)

        depth = 128
        new_image_features = feature_pyramid_network(image_features[:5], is_training, depth)

        return new_image_features + image_features[5:]


def feature_pyramid_network(image_features, is_training, depth):

    def batch_norm(x):
        x = tf.layers.batch_normalization(
            x, axis=1, center=True, scale=True,
            momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
            training=is_training, fused=True,
            name='BatchNorm'
        )
        return x

    params = {
        'padding': 'SAME',
        'activation_fn': tf.nn.relu6,
        'normalizer_fn': batch_norm,
        'data_format': 'NCHW'
    }
    with slim.arg_scope([slim.conv2d], **params):

        top_down = slim.conv2d(image_features[-1], depth, [1, 1], scope='fpn_beginning')
        new_image_features.append(top_down)

        for i, feature_map in reversed(enumerate(image_features[:-1])):
            with tf.variable_scope('fpn_block_%d' % i):
                new_feature_map, top_down = fpn_block(feature_map, top_down, depth)
                new_image_features.append(new_feature_map)

    return reversed(new_image_features)


def fpn_block(feature_map, top_down, depth):
    if not is_same_size(feature_map, top_down):
        top_down = nearest_neighbor_upsampling(top_down)
    residual = slim.conv2d(feature_map, depth, [1, 1], scope='Conv2d_1x1')
    top_down = 0.5*(top_down + residual)
    new_feature_map = slim.conv2d(top_down, depth, [3, 3], scope='Conv2d_3x3')
    return new_feature_map, top_down


def is_same_size(x, y):
    height1, width1 = x.shape.as_list()[2:]
    height2, width2 = y.shape.as_list()[2:]
    return (height1 == height2) and (width1 == width2)


def nearest_neighbor_upsampling(x, scale=2):
    with tf.name_scope('upsampling'):

        batch_size = tf.shape(x)[0]
        channels, height, width = x.shape.as_list()[1:]
        shape_before_tile = [batch_size, channels, height, 1, width, 1]
        shape_after_tile = [batch_size, channels, height * scale, width * scale]

        x = tf.reshape(x, shape_before_tile)
        x = tf.tile(x, [1, 1, 1, scale, 1, scale])
        x = tf.reshape(x, shape_after_tile)
        return x
