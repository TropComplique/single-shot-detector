from abc import ABC, abstractmethod
import tensorflow as tf
from .constants import DATA_FORMAT, MIN_LEVEL
from .utils import batch_norm_relu, conv2d_same


FPN_DEPTH = 256


class FeatureExtractor(ABC):

    def __init__(self, is_training, backbone):
        self.is_training = is_training
        self.backbone = backbone

    @abstractmethod
    def __call__(self, images):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, channels_i, height_i, width_i].
        """
        pass


class RetinaNetFeatureExtractor(FeatureExtractor):

    def __call__(self, images):
        features = self.backbone(images, self.is_training)
        enriched_features = fpn(
            features, self.is_training,
            min_level=MIN_LEVEL, scope='fpn'
        )
        return [enriched_features['p' + str(i)] for i in range(MIN_LEVEL, 8)]


def fpn(features, is_training, min_level=3, scope='fpn'):
    """
    Arguments:
        features: a dict with three float tensors.
            It must have keys ['c3', 'c4', 'c5'].
            Where a number means that a feature has stride `2**number`.
        is_training: a boolean.
        min_level: an integer, minimal feature stride
            that will be used is `2**min_level`.
            Possible values are [3, 4, 5]
        scope: a string.
    Returns:
        a dict with float tensors.
    """

    with tf.variable_scope(scope):

        x = conv2d_same(features['c5'], FPN_DEPTH, kernel_size=1, name='lateral5')
        p5 = conv2d_same(x, FPN_DEPTH, kernel_size=3, name='p5')
        p6 = conv2d_same(features['c5'], FPN_DEPTH, kernel_size=3, stride=2, name='p6')
        p7 = conv2d_same(tf.nn.relu(p6), FPN_DEPTH, kernel_size=3, stride=2, name='p7')
        enriched_features = {'p5': p5, 'p6': p6, 'p7': p7}

        # top-down path
        for i in reversed(range(min_level, 5)):
            i = str(i)
            lateral = conv2d_same(features['c' + i], FPN_DEPTH, kernel_size=1, name='lateral' + i)
            x = nearest_neighbor_upsample(x, scope='upsampling' + i) + lateral
            p = conv2d_same(x, FPN_DEPTH, kernel_size=3, name='p' + i)
            enriched_features['p' + i] = p

        enriched_features = {
            n: batch_norm_relu(x, is_training, use_relu=False, name=n + '_batch_norm')
            for n, x in enriched_features.items()
        }

    return enriched_features


def nearest_neighbor_upsample(x, rate=2, scope='upsampling'):
    with tf.name_scope(scope):

        shape = tf.shape(x)
        batch_size = x.shape[0].value
        if batch_size is None:
            batch_size = shape[0]

        if DATA_FORMAT == 'channels_first':
            channels = x.shape[1].value
            height, width = shape[2], shape[3]
            x = tf.reshape(x, [batch_size, channels, height, 1, width, 1])
            x = tf.tile(x, [1, 1, 1, rate, 1, rate])
            x = tf.reshape(x, [batch_size, channels, height * rate, width * rate])
        else:
            height, width = shape[1], shape[2]
            channels = x.shape[3].value
            x = tf.reshape(x, [batch_size, height, 1, width, 1, channels])
            x = tf.tile(x, [1, 1, rate, 1, rate, 1])
            x = tf.reshape(x, [batch_size, height * rate, width * rate, channels])

        return x


class SSDFeatureExtractor(FeatureExtractor):

    def __call__(self, images):

        features = self.backbone(images, self.is_training)
        image_features = [features['c4'], features['c5']]
        x = features['c5']

        filters = [512, 256, 256, 128]
        for i, num_filters in enumerate(filters):
            x = conv2d_same(x, num_filters // 2, kernel_size=1, name='conv1_%d' % i)
            x = batch_norm_relu(x, self.is_training, name='batch_norm1_%d' % i)
            x = conv2d_same(x, num_filters, kernel_size=3, stride=2, name='conv2_%d' % i)
            x = batch_norm_relu(x, self.is_training, name='batch_norm2_%d' % i)
            image_features.append(x)

        return image_features
