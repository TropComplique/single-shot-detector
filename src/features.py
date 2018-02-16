import tensorflow as tf
import tensorflow.contrib.slim as slim


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
                momentum=0.9, epsilon=0.001, 
                training=self.is_training, fused=True,
                name='BatchNorm'
            )
            return x

        filters = [192, 192, 96, 96]
        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }
        with slim.arg_scope([slim.conv2d], **params):
            for i, num_filters in enumerate(filters, 14):
                x = slim.conv2d(x, num_filters // 2, (1, 1), stride=1, scope='Conv2d_%d' % i)
                x = slim.conv2d(x, num_filters, (3, 3), stride=2, scope='Conv2d_%d_1x1' % i)
                image_features.append(x)

        return image_features
