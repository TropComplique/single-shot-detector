import tensorflow as tf
import tensorflow.contrib.slim as slim
from .constants import DATA_FORMAT, MATCHING_THRESHOLD
from .utils import batch_multiclass_non_max_suppression, batch_decode
from .training_target_creation import get_targets
from .losses import localization_loss, classification_loss, apply_hard_mining


class SSD:
    def __init__(self, images, feature_extractor, anchor_generator, num_classes):

        feature_maps = feature_extractor(images)
        self.num_classes = num_classes
        self.anchors = anchor_generator.generate(feature_maps)
        self.num_basis_anchors = anchor_generator.num_basis_anchors
        self._add_box_predictions(feature_maps)

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
        num_basis_anchors = self.num_basis_anchors
        num_feature_maps = len(feature_maps)

        box_encodings, class_predictions_with_background = [], []
        for i, x, num_predictions_per_location in zip(range(num_feature_maps), feature_maps, num_basis_anchors):

            batch_size = tf.shape(x)[0]
            _, _, height_i, width_i = x.shape.as_list()
            num_anchors_on_feature_map = height_i * width_i * num_predictions_per_location

            y = slim.conv2d(
                x, num_predictions_per_location * 4,
                [1, 1], activation_fn=None, scope='box_encoding_predictor_%d' % i,
                data_format=DATA_FORMAT
            )
            # it has shape [batch_size, num_predictions_per_location * 4, height_i, width_i]
            y = tf.transpose(y, perm=[0, 2, 3, 1])
            y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, 4])
            box_encodings.append(tf.reshape(y, [batch_size, num_anchors_on_feature_map, 4]))

            y = slim.conv2d(
                x, num_predictions_per_location * (num_classes + 1),
                [1, 1], activation_fn=None, scope='class_predictor_%d' % i,
                data_format=DATA_FORMAT
            )
            # it has  shape [batch_size, num_predictions_per_location * (num_classes + 1), height_i, width_i]
            y = tf.transpose(y, perm=[0, 2, 3, 1])
            y = tf.reshape(y, [batch_size, height_i, width_i, num_predictions_per_location, num_classes + 1])
            class_predictions_with_background.append(tf.reshape(y, [batch_size, num_anchors_on_feature_map, num_classes + 1]))

        self.box_encodings = tf.concat(box_encodings, axis=1)
        self.class_predictions_with_background = tf.concat(class_predictions_with_background, axis=1)

    def get_predictions(self, score_threshold=0.1, iou_threshold=0.6, max_boxes_per_class=20):
        with tf.name_scope('postprocessing'):
            boxes = batch_decode(self.box_encodings, self.anchors)
            class_predictions_without_background = tf.slice(
                self.class_predictions_with_background,
                [0, 0, 1], [-1, -1, -1]
            )
            scores = tf.sigmoid(class_predictions_without_background)

            boxes, scores, classes, num_detections = batch_multiclass_non_max_suppression(
                boxes, scores,
                score_threshold, iou_threshold,
                max_boxes_per_class, self.num_classes
            )
            return boxes, scores, classes, num_detections

    def loss(self, boxes, labels, num_boxes):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            boxes: a float tensor with shape [batch_size, max_num_boxes, 4].
            labels: an int tensor with shape [batch_size, max_num_boxes].
            num_boxes: an int tensor with shape [batch_size].
                where max_num_boxes = max(num_boxes).
        Returns:
            a dict with two keys ('localization_loss' and 'classification_loss')
            which contains float tensors with shape [].
        """

        cls_targets, reg_targets, matches = self._create_targets(boxes, labels, num_boxes)

        match_indicator = tf.greater_equal(matches, 0)
        weights = tf.to_float(match_indicator)
        num_matches = tf.reduce_sum(weights)  # shape []

        location_losses = localization_loss(
            self.box_encodings,
            reg_targets,
            weights
        )  # shape: [batch_size, num_anchors]
        cls_losses = classification_loss(
            self.class_predictions_with_background,
            cls_targets,
            weights
        )  # shape: [batch_size, num_anchor]

        location_loss, cls_loss = apply_hard_mining(
            location_losses, cls_losses, self.class_predictions_with_background,
            self.box_encodings, matches, self.anchors
        )

        normalizer = tf.maximum(num_matches, 1.0)
        localization_loss_weight = 1.0
        classification_loss_weight = 1.0
        location_loss = (localization_loss_weight / normalizer) * location_loss
        cls_loss = (classification_loss_weight / normalizer) * cls_loss
        return location_loss + cls_loss

    def _create_targets(self, boxes, labels, num_boxes):
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

        cls_targets, reg_targets, matches = tf.map_fn(
            fn,
            [boxes, labels, num_boxes],
            dtype=(tf.float32, tf.float32, tf.int32),
            parallel_iterations=10,
            back_prop=False,
            swap_memory=False,
            infer_shape=True
        )
        return cls_targets, reg_targets, matches
