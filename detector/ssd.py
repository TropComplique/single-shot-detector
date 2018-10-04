import tensorflow as tf

from detector.constants import PARALLEL_ITERATIONS, POSITIVES_THRESHOLD, NEGATIVES_THRESHOLD, MIN_LEVEL
from detector.utils import batch_multiclass_non_max_suppression
from detector.training_target_creation import get_training_targets
from detector.losses import localization_loss, usual_classification_loss, focal_loss, apply_hard_mining


class SSD:
    def __init__(self, images, feature_extractor, anchor_generator, box_predictor, num_classes):
        """
        Arguments:
            images: a float tensor with shape [batch_size, height, width, 3],
                a batch of RGB images with pixel values in the range [0, 1].
            feature_extractor: an instance of FeatureExtractor.
            anchor_generator: an instance of AnchorGenerator.
            box_predictor: an instance of BoxPredictor.
            num_classes: an integer, number of labels without
                counting background class.
        """
        self.num_classes = num_classes

        image_features = feature_extractor(images)
        # `image_features` a list of float tensors where the ith tensor
        # has shape [batch_size, channels_i, height_i, width_i].

        # the detector supports images of various sizes
        shape = tf.shape(images)
        image_height, image_width = shape[1], shape[2]

        self.anchors = anchor_generator(image_height, image_width)
        # it has shape [num_anchors, 4]

        # this is used for summaries only
        self.num_anchors_per_feature_map = anchor_generator.num_anchors_per_feature_map

        self.raw_predictions = box_predictor(image_features)
        # a dict with two float tensors:
        # `encoded_boxes` has shape [batch_size, num_anchors, num_classes, 4],
        # `class_predictions` has shape [batch_size, num_anchors, num_classes]

    def get_predictions(self, score_threshold=0.05, iou_threshold=0.5, max_boxes_per_class=20):
        """Postprocess outputs of the network.

        Returns:
            boxes: a float tensor with shape [batch_size, N, 4].
            labels: an int tensor with shape [batch_size, N].
            scores: a float tensor with shape [batch_size, N].
            num_boxes: an int tensor with shape [batch_size], it
                represents the number of detections on an image.

            Where N = num_classes * max_boxes_per_class.
        """
        with tf.name_scope('postprocessing'):

            encoded_boxes = self.raw_predictions['encoded_boxes']
            # it has shape [batch_size, num_anchors, num_classes, 4]

            class_predictions = self.raw_predictions['class_predictions']
            scores = tf.sigmoid(class_predictions)
            # it has shape [batch_size, num_anchors, num_classes]

        with tf.name_scope('nms'):
            boxes, scores, classes, num_detections = batch_multiclass_non_max_suppression(
                encoded_boxes, self.anchors, scores,
                score_threshold=score_threshold, iou_threshold=iou_threshold,
                max_boxes_per_class=max_boxes_per_class
            )
        return {'boxes': boxes, 'labels': classes, 'scores': scores, 'num_boxes': num_detections}

    def loss(self, groundtruth, params):
        """Compute scalar loss tensors with respect to provided groundtruth.

        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'labels': an int tensor with shape [batch_size, max_num_boxes].
                'num_boxes': an int tensor with shape [batch_size],
                    where max_num_boxes = max(num_boxes).
            params: a dict with parameters.
        Returns:
            two float tensors with shape [].
        """
        reg_targets, cls_targets, matches = self._create_targets(groundtruth)

        with tf.name_scope('losses'):

            # whether anchor is matched
            weights = tf.to_float(tf.greater_equal(matches, 0))

            with tf.name_scope('classification_loss'):

                class_predictions = self.raw_predictions['class_predictions']
                # shape [batch_size, num_anchors, num_classes]

                cls_targets = tf.one_hot(cls_targets, self.num_classes + 1, axis=2)
                # shape [batch_size, num_anchors, num_classes + 1]

                # remove background
                cls_targets = tf.to_float(cls_targets[:, :, 1:])
                # now background represented by all zeros

                not_ignore = tf.to_float(tf.greater_equal(matches, -1))
                # if a value is `-2` then we ignore its anchor

                if params['use_focal_loss']:
                    cls_losses = focal_loss(
                        class_predictions, cls_targets, weights=not_ignore,
                        gamma=params['gamma'], alpha=params['alpha']
                    )
                else:
                    cls_losses = usual_classification_loss(
                        class_predictions, cls_targets,
                        weights=not_ignore
                    )
                # `cls_losses` has shape [batch_size, num_anchors]

            with tf.name_scope('localization_loss'):

                encoded_boxes = self.raw_predictions['encoded_boxes']
                # it has shape [batch_size, num_anchors, num_classes, 4]

                # choose only boxes of a true class
                cls_targets = tf.expand_dims(cls_targets, axis=3)
                encoded_boxes *= cls_targets
                encoded_boxes = tf.reduce_sum(encoded_boxes, axis=2)
                # it has shape [batch_size, num_anchors, 4]

                loc_losses = localization_loss(encoded_boxes, reg_targets, weights)
                # shape [batch_size, num_anchors]

            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)

            with tf.name_scope('loss_summaries'):
                self._add_scalewise_matches_summaries(weights)
                self._add_scalewise_summaries(cls_losses, name='classification_losses')
                self._add_scalewise_summaries(loc_losses, name='localization_losses')
                tf.summary.scalar('total_mean_matches_per_image', tf.reduce_mean(matches_per_image))

            if params['use_ohem']:
                with tf.name_scope('ohem'):
                    loc_loss, cls_loss = apply_hard_mining(
                        loc_losses, cls_losses,
                        encoded_boxes, matches, self.anchors,
                        loss_to_use=params['loss_to_use'],
                        loc_loss_weight=params['loc_loss_weight'],
                        cls_loss_weight=params['cls_loss_weight'],
                        num_hard_examples=params['num_hard_examples'],
                        nms_threshold=params['nms_threshold'],
                        max_negatives_per_positive=params['max_negatives_per_positive'],
                        min_negatives_per_image=params['min_negatives_per_image']
                    )
            else:
                loc_loss = tf.reduce_sum(loc_losses, axis=[0, 1])
                cls_loss = tf.reduce_sum(cls_losses, axis=[0, 1])

            return {'localization_loss': loc_loss/normalizer, 'classification_loss': cls_loss/normalizer}

    def _add_scalewise_summaries(self, tensor, name):
        """Adds histograms of the biggest 20 percent of
        tensor's values for each scale (feature map).

        Arguments:
            tensor: a float tensor with shape [batch_size, num_anchors].
            name: a string.
        """
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map, MIN_LEVEL):
            k = tf.to_int32(tf.ceil(tf.to_float(n) * 0.20))  # top 20%
            biggest_values, _ = tf.nn.top_k(tensor[:, index:(index + n)], k, sorted=False)
            # it has shape [batch_size, k]
            tf.summary.histogram(
                name + '_on_level_' + str(i),
                tf.reduce_mean(biggest_values, axis=0)
            )
            index += n

    def _add_scalewise_matches_summaries(self, weights):
        """Adds summaries for the number of matches on each scale."""
        index = 0
        for i, n in enumerate(self.num_anchors_per_feature_map, MIN_LEVEL):
            matches_per_image = tf.reduce_sum(weights[:, index:(index + n)], axis=1)
            tf.summary.scalar(
                'mean_matches_per_image_on_level_' + str(i),
                tf.reduce_mean(matches_per_image, axis=0)
            )
            index += n

    def _create_targets(self, groundtruth):
        """
        Arguments:
            groundtruth: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, N, 4].
                'labels': an int tensor with shape [batch_size, N].
                'num_boxes': an int tensor with shape [batch_size].
        Returns:
            reg_targets: a float tensor with shape [batch_size, num_anchors, 4].
            cls_targets: an int tensor with shape [batch_size, num_anchors],
                possible values that it can contain are [0, 1, 2, ..., num_classes],
                `0` is a background class.
            matches: an int tensor with shape [batch_size, num_anchors],
                `-1` means that an anchor box is negative (background),
                and `-2` means that we must ignore this anchor box.
        """
        def fn(x):
            boxes, labels, num_boxes = x
            boxes, labels = boxes[:num_boxes], labels[:num_boxes]

            reg_targets, cls_targets, matches = get_training_targets(
                self.anchors, boxes, labels,
                positives_threshold=POSITIVES_THRESHOLD,
                negatives_threshold=NEGATIVES_THRESHOLD
            )
            return reg_targets, cls_targets, matches

        with tf.name_scope('target_creation'):
            reg_targets, cls_targets, matches = tf.map_fn(
                fn, [groundtruth['boxes'], groundtruth['labels'], groundtruth['num_boxes']],
                dtype=(tf.float32, tf.int32, tf.int32),
                parallel_iterations=PARALLEL_ITERATIONS,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return reg_targets, cls_targets, matches
