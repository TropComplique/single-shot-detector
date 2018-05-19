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
                a batch of RGB images with pixel values in the range [0, 1].
        Returns:
            a list of float tensors where the ith tensor
            has shape [batch, channels_i, height_i, width_i].
        """

        x, feature_maps = self.backbone(images, self.is_training)
        image_features = [
            feature_maps['expanded_conv_12'],  # scale 0
            feature_maps['Conv_1']  # scale 1
        ]

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=1, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=self.is_training, fused=True,
                name='BatchNorm'
            )
            return x

        # scales 2, 3, 4, 5:
        filters = [512, 256, 256, 128]
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }
        with slim.arg_scope([slim.conv2d], **params):
            for i, num_filters in enumerate(filters, 2):
                x = slim.conv2d(x, num_filters // 2, (1, 1), stride=1, scope='Conv2d_scale_%d_1x1' % i)
                x = slim.conv2d(x, num_filters, (3, 3), stride=2, scope='Conv2d_scale_%d' % i)
                image_features.append(x)

        return image_features
