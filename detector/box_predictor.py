from abc import ABC, abstractmethod
import math
import tensorflow as tf
from .constants import DATA_FORMAT, MIN_LEVEL
from .utils import batch_norm_relu, conv2d_same


BOX_PREDICTOR_DEPTH = 64


class BoxPredictor(ABC):

    def __init__(self, is_training, num_classes, num_anchors_per_location):
        self.is_training = is_training
        self.num_classes = num_classes
        self.num_anchors_per_location = num_anchors_per_location

    @abstractmethod
    def __call__(self, image_features):
        """
        Adds box predictors to each feature map,
        reshapes, and returns concatenated results.

        Arguments:
            image_features: a list of float tensors where the ith tensor
                has shape [batch_size, channels_i, height_i, width_i].
        Returns:
            encoded_boxes: a float tensor with shape [batch_size, num_anchors, num_classes, 4].
            class_predictions: a float tensor with shape [batch_size, num_anchors, num_classes].
        """
        pass


class SSDBoxPredictor(BoxPredictor):

    def __call__(self, image_features):

        encoded_boxes = []
        class_predictions = []

        with tf.variable_scope('prediction_layers'):

            for i in range(len(image_features)):

                x = image_features[i]
                y = conv2d_same(
                    x, self.num_classes * self.num_anchors_per_location * 4,
                    kernel_size=1, name='box_encoding_predictor_%d' % i
                )
                # it has shape [batch_size, num_classes * num_anchors_per_location * 4, height_i, width_i]
                encoded_boxes.append(y)

                y = conv2d_same(
                    x, self.num_classes * self.num_anchors_per_location,
                    kernel_size=1, name='class_predictor_%d' % i
                )
                # it has  shape [batch_size, num_classes * num_anchors_per_location, height_i, width_i]
                class_predictions.append(y)

        return reshape_and_concatenate(
            encoded_boxes, class_predictions,
            self.num_classes, self.num_anchors_per_location
        )


class RetinaNetBoxPredictor(BoxPredictor):

    def __call__(self, image_features):

        encoded_boxes = []
        class_predictions = []

        """
        The convolution layers in the box net are shared among all levels, but
        each level has its batch normalization to capture the statistical
        difference among different levels. The same for the class net.
        """

        with tf.variable_scope('box_net', reuse=tf.AUTO_REUSE):
            for level, p in enumerate(image_features, MIN_LEVEL):
                encoded_boxes.append(box_net(
                    p, self.is_training, level, self.num_classes,
                    self.num_anchors_per_location
                ))

        with tf.variable_scope('class_net', reuse=tf.AUTO_REUSE):
            for level, p in enumerate(image_features, MIN_LEVEL):
                class_predictions.append(class_net(
                    p, self.is_training, level, self.num_classes,
                    self.num_anchors_per_location
                ))

        return reshape_and_concatenate(
            encoded_boxes, class_predictions,
            self.num_classes, self.num_anchors_per_location
        )


def reshape_and_concatenate(
        encoded_boxes, class_predictions,
        num_classes, num_anchors_per_location):
    
    # batch size is a static value 
    # during training and evaluation
    batch_size = encoded_boxes[0].shape[0].value
    if batch_size is None:
        batch_size = tf.shape(encoded_boxes[0])[0]
                
    # it is important that reshaping here is the same as when anchors were generated
    with tf.name_scope('reshaping_and_concatenation'):
        for i in range(len(encoded_boxes)):

            # get spatial dimensions
            shape = tf.shape(encoded_boxes[i])
            if DATA_FORMAT == 'channels_first':
                height_i, width_i = shape[2], shape[3]
            else:
                height_i, width_i = shape[1], shape[2]

            # total number of anchors
            num_anchors_on_feature_map = height_i * width_i * num_anchors_per_location

            y = encoded_boxes[i]
            y = tf.transpose(y, perm=[0, 2, 3, 1]) if DATA_FORMAT == 'channels_first' else y
            y = tf.reshape(y, [batch_size, height_i, width_i, num_anchors_per_location, num_classes, 4])
            encoded_boxes[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map, num_classes, 4])

            y = class_predictions[i]
            y = tf.transpose(y, perm=[0, 2, 3, 1]) if DATA_FORMAT == 'channels_first' else y
            y = tf.reshape(y, [batch_size, height_i, width_i, num_anchors_per_location, num_classes])
            class_predictions[i] = tf.reshape(y, [batch_size, num_anchors_on_feature_map, num_classes])

        encoded_boxes = tf.concat(encoded_boxes, axis=1)
        class_predictions = tf.concat(class_predictions, axis=1)

    return {'encoded_boxes': encoded_boxes, 'class_predictions': class_predictions}


def class_net(x, is_training, level, num_classes, num_anchors_per_location):
    """
    Arguments:
        x: a float tensor with shape [batch_size, depth, height, width].
        is_training: a boolean.
        level, num_classes, num_anchors_per_location: integers.
    Returns:
        a float tensor with shape [batch_size, num_classes * num_anchors_per_location, height, width].
    """

    for i in range(4):
        x = conv2d_same(x, BOX_PREDICTOR_DEPTH, kernel_size=3, name='conv3x3_%d' % i)
        x = batch_norm_relu(x, is_training, name='batch_norm_%d_for_level_%d' % (i, level))

    p = 0.01  # probability of foreground
    # sigmoid(-log((1 - p) / p)) = p

    logits = tf.layers.conv2d(
        x, num_classes * num_anchors_per_location,
        kernel_size=(3, 3), padding='same',
        bias_initializer=tf.constant_initializer(-math.log((1.0 - p) / p)),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        data_format=DATA_FORMAT, name='logits'
    )
    return logits


def box_net(x, is_training, level, num_classes, num_anchors_per_location):
    """
    Arguments:
        x: a float tensor with shape [batch_size, depth, height, width].
        is_training: a boolean.
        level, num_classes, num_anchors_per_location: integers.
    Returns:
        a float tensor with shape [batch_size, 4 * num_classes * num_anchors_per_location, height, width].
    """

    for i in range(4):
        x = conv2d_same(x, BOX_PREDICTOR_DEPTH, kernel_size=3, name='conv3x3_%d' % i)
        x = batch_norm_relu(x, is_training, name='batch_norm_%d_for_level_%d' % (i, level))

    encoded_boxes = tf.layers.conv2d(
        x, 4 * num_classes * num_anchors_per_location,
        kernel_size=(3, 3), padding='same',
        bias_initializer=tf.zeros_initializer(),
        kernel_initializer=tf.random_normal_initializer(stddev=0.01),
        data_format=DATA_FORMAT, name='encoded_boxes'
    )
    return encoded_boxes
