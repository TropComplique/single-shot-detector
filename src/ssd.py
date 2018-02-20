import tensorflow as tf
import tensorflow.contrib.slim as slim

from .constants import MATCHING_THRESHOLD, PARALLEL_ITERATIONS, BATCH_NORM_MOMENTUM
from .utils import batch_multiclass_non_max_suppression, batch_decode
from .training_target_creation import get_targets
from .losses import localization_loss, classification_loss, apply_hard_mining


class SSD:
    def __init__(self, images, feature_extractor, anchor_generator, num_classes):

        feature_maps = feature_extractor(images)
        self.is_training = feature_extractor.is_training
        self.num_classes = num_classes
        with tf.name_scope('anchor_generator'):
            self.anchors = anchor_generator(feature_maps, images)
        self.num_anchors_per_location = anchor_generator.num_anchors_per_location
        with tf.name_scope('prediction_layers'):
            self._add_box_predictions(feature_maps)

    def get_predictions(self, score_threshold=0.1, iou_threshold=0.6, max_boxes_per_class=20):
        with tf.name_scope('postprocessing'):
            boxes = batch_decode(self.box_encodings, self.anchors)

            scores = tf.nn.softmax(self.class_predictions_with_background)
            scores = tf.slice(scores, [0, 0, 1], [-1, -1, -1])

            # class_predictions_without_background = tf.slice(
            #     self.class_predictions_with_background,
            #     [0, 0, 1], [-1, -1, -1]
            # )
            # scores = tf.sigmoid(class_predictions_without_background)

            boxes, scores, classes, num_detections = batch_multiclass_non_max_suppression(
                boxes, scores,
                score_threshold, iou_threshold,
                max_boxes_per_class, self.num_classes
            )
            return {'boxes': boxes, 'labels': classes, 'scores': scores, 'num_boxes': num_detections}

    def loss(self, groundtruth, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'labels': an int tensor with shape [batch_size, max_num_boxes].
                'num_boxes': an int tensor with shape [batch_size].
                    where max_num_boxes = max(num_boxes).
            params: a dict with parameters for losses.
        Returns:
            two float tensors with shape [].
        """

        cls_targets, reg_targets, matches = self._create_targets(groundtruth)

        with tf.name_scope('losses'):
            weights = tf.to_float(tf.greater_equal(matches, 0))
            matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
            tf.summary.scalar('mean_matches_per_image', tf.reduce_mean(matches_per_image))
            num_matches = tf.reduce_sum(matches_per_image)  # shape []

            with tf.name_scope('classification_loss'):
                cls_losses = classification_loss(
                    self.class_predictions_with_background,
                    cls_targets
                )
            with tf.name_scope('localization_loss'):
                location_losses = localization_loss(
                    self.box_encodings,
                    reg_targets, weights
                )
            # they have shape [batch_size, num_anchors]

            tf.summary.histogram('all_classification_losses', cls_losses)
            tf.summary.histogram('all_localization_losses', location_losses)

            with tf.name_scope('ohem'):
                location_loss, cls_loss = apply_hard_mining(
                    location_losses, cls_losses,
                    self.class_predictions_with_background,
                    self.box_encodings, matches, self.anchors,
                    loss_to_use=params['loss_to_use'],
                    loc_loss_weight=params['loc_loss_weight'],
                    cls_loss_weight=params['cls_loss_weight'],
                    num_hard_examples=params['num_hard_examples'],
                    nms_threshold=params['nms_threshold'],
                    max_negatives_per_positive=params['max_negatives_per_positive'],
                    min_negatives_per_image=params['min_negatives_per_image']
                )
            normalizer = tf.maximum(num_matches, 1.0)
            return {'localization_loss': location_loss/normalizer, 'classification_loss': cls_loss/normalizer}

    def _create_targets(self, groundtruth):
        """
        Arguments:
            boxes: a float tensor with shape [batch_size, N, 4].
            labels: an int tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size].
        Returns:
            cls_targets: a float tensor with shape [batch_size, num_anchors, num_classes + 1].
            reg_targets: a float tensor with shape [batch_size, num_anchors, 4].
            matches: an int tensor with shape [batch_size, num_anchors].
        """
        def fn(x):
            boxes, labels, num_boxes = x
            boxes, labels = boxes[:num_boxes], labels[:num_boxes]
            labels = tf.one_hot(labels, self.num_classes, axis=1, dtype=tf.float32)
            cls_targets, reg_targets, matches = get_targets(
                self.anchors, boxes, labels,
                self.num_classes, threshold=MATCHING_THRESHOLD
            )
            return cls_targets, reg_targets, matches

        with tf.name_scope('target_creation'):
            cls_targets, reg_targets, matches = tf.map_fn(
                fn, [groundtruth['boxes'], groundtruth['labels'], groundtruth['num_boxes']],
                dtype=(tf.float32, tf.float32, tf.int32),
                parallel_iterations=PARALLEL_ITERATIONS,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return cls_targets, reg_targets, matches

    def _add_box_predictions(self, feature_maps):
        """Adds box predictors to each feature map and returns concatenated results.

        Arguments:
            feature_maps: a list of float tensors where the ith tensor has shape
                [batch, channels_i, height_i, width_i].

        It creates two tensors:
            box_encodings: a float tensor with shape [batch_size, num_anchors, 4].
            class_predictions_with_background: a float tensor with shape
                [batch_size, num_anchors, num_classes + 1].
        """
        num_classes = self.num_classes
        num_anchors_per_location = self.num_anchors_per_location
        num_feature_maps = len(feature_maps)

        def batch_norm(x):
            x = tf.layers.batch_normalization(
                x, axis=1, center=True, scale=True,
                momentum=BATCH_NORM_MOMENTUM, epsilon=0.001,
                training=self.is_training, fused=True,
                name='BatchNorm'
            )
            return x

        params = {
            'padding': 'SAME',
            'activation_fn': tf.nn.relu6,
            'normalizer_fn': batch_norm,
            'data_format': 'NCHW'
        }

        box_encodings, class_predictions_with_background = [], []
        for i, x, num_predictions_per_location in zip(range(num_feature_maps), feature_maps, num_anchors_per_location):

            batch_size = tf.shape(x)[0]
            height_i, width_i = x.shape.as_list()[2:]
            num_anchors_on_feature_map = height_i * width_i * num_predictions_per_location
            
            with slim.arg_scope([slim.conv2d], **params):
                x1 = slim.conv2d(
                    x, 32,
                    [1, 1], scope='box_encoding_predictor1',
                    reuse=tf.AUTO_REUSE,
                    data_format='NCHW'
                )
            y = slim.conv2d(
                x1, num_predictions_per_location * 4,
                [1, 1], activation_fn=None, scope='box_encoding_predictor2',
                reuse=tf.AUTO_REUSE,
                data_format='NCHW'
            )
            # it has shape [batch_size, num_predictions_per_location * 4, height_i, width_i]
            with tf.name_scope('reshape_predictions_%d' % i):
                y = tf.transpose(y, perm=[0, 2, 3, 1])
                y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, 4])
                box_encodings.append(tf.reshape(y, [batch_size, num_anchors_on_feature_map, 4]))

            y = slim.conv2d(
                x, num_predictions_per_location * (num_classes + 1),
                [1, 1], activation_fn=None, scope='class_predictor',
                reuse=tf.AUTO_REUSE,
                data_format='NCHW'
            )
            # it has  shape [batch_size, num_predictions_per_location * (num_classes + 1), height_i, width_i]
            with tf.name_scope('reshape_predictions_%d' % i):
                y = tf.transpose(y, perm=[0, 2, 3, 1])
                y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, num_classes + 1])
                class_predictions_with_background.append(tf.reshape(y, [batch_size, num_anchors_on_feature_map, num_classes + 1]))

        with tf.name_scope('concat_predictions'):
            self.box_encodings = tf.concat(box_encodings, axis=1)
            self.class_predictions_with_background = tf.concat(class_predictions_with_background, axis=1)
