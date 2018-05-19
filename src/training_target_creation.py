import tensorflow as tf
from src.utils import encode, iou


def get_training_targets(anchors, groundtruth_boxes, groundtruth_labels, num_classes, threshold=0.5):
    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        groundtruth_labels: a float tensor with shape [N, num_classes].
        num_classes: an integer.
        threshold: a float number.
    Returns:
        cls_targets: a float tensor with shape [num_anchors, num_classes + 1].
        reg_targets: a float tensor with shape [num_anchors, 4].
        matches: an int tensor with shape [num_anchors], possible values
            that it can contain are [-1, 0, 1, 2, ..., (N - 1)].
    """
    with tf.name_scope('matching'):
        N = tf.shape(groundtruth_boxes)[0]
        num_anchors = anchors.shape[0]
        matches = tf.cond(
            tf.greater(N, 0),
            lambda: _match(anchors, groundtruth_boxes, threshold),
            lambda: -1 * tf.ones([num_anchors], dtype=tf.int32)
        )
        matches = tf.to_int32(matches)

    with tf.name_scope('reg_and_cls_target_creation'):
        reg_targets, cls_targets = _create_targets(
            anchors, groundtruth_boxes, groundtruth_labels,
            matches, num_classes
        )

    return cls_targets, reg_targets, matches


def _match(anchors, groundtruth_boxes, threshold=0.5):
    """Matching algorithm:
    1) for each groundtruth box choose the anchor with largest iou,
    2) remove this set of anchors from the set of all anchors,
    3) for each remaining anchor choose the groundtruth box with largest iou,
       but only if this iou is larger than `threshold`.

    Note: after step 1, it could happen that for some two groundtruth boxes
    chosen anchors are the same. Let's hope this never happens.
    Also see the comments below.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        an int tensor with shape [num_anchors].
    """
    num_anchors = anchors.shape.as_list()[0]

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(groundtruth_boxes, anchors)  # shape [N, num_anchors]
    matches = tf.argmax(similarity_matrix, axis=0, output_type=tf.int32)  # shape [num_anchors]
    matched_vals = tf.reduce_max(similarity_matrix, axis=0)  # shape [num_anchors]
    below_threshold = tf.to_int32(tf.greater(threshold, matched_vals))
    matches = tf.add(tf.multiply(matches, 1 - below_threshold), -1 * below_threshold)
    # after this, it could happen that some groundtruth
    # boxes are not matched with any anchor box

    # now we must ensure that each row (groundtruth box) is matched to
    # at least one column (which is not guaranteed
    # otherwise if `threshold` is high)

    # for each groundtruth box choose the anchor box with largest iou
    # (force match for each groundtruth box)
    forced_matches_ids = tf.argmax(similarity_matrix, axis=1, output_type=tf.int32)  # shape [N]
    # if all indices in forced_matches_ids are different then all rows will be matched

    forced_matches_indicators = tf.one_hot(forced_matches_ids, depth=num_anchors, dtype=tf.int32)  # shape [N, num_anchors]
    forced_match_row_ids = tf.argmax(forced_matches_indicators, axis=0, output_type=tf.int32)  # shape [num_anchors]
    forced_match_mask = tf.greater(tf.reduce_max(forced_matches_indicators, axis=0), 0)  # shape [num_anchors]
    matches = tf.where(forced_match_mask, forced_match_row_ids, matches)
    # even after this it could happen that some rows aren't matched,
    # but i believe that this event has low probability

    return matches


def _create_targets(anchors, groundtruth_boxes, groundtruth_labels, matches, num_classes):
    """Returns regression and classification targets for each anchor.

    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        groundtruth_labels: a float tensor with shape [N, num_classes].
        matches: a int tensor with shape [num_anchors].
        num_classes: an integer.
    Returns:
        cls_targets: a float tensor with shape [num_anchors, num_classes + 1].
        reg_targets: a float tensor with shape [num_anchors, 4].
    """
    num_anchors = anchors.shape.as_list()[0]

    matched_anchor_indices = tf.where(tf.greater_equal(matches, 0))  # shape [num_matches, 1]
    matched_anchor_indices = tf.squeeze(matched_anchor_indices, axis=1)
    matched_gt_indices = tf.gather(matches, matched_anchor_indices)  # shape [num_matches]

    matched_anchors = tf.gather(anchors, matched_anchor_indices)  # shape [num_matches, 4]
    matched_gt_boxes = tf.gather(groundtruth_boxes, matched_gt_indices)  # shape [num_matches, 4]
    matched_reg_targets = encode(matched_gt_boxes, matched_anchors)  # shape [num_matches, 4]

    unmatched_anchor_indices = tf.where(tf.equal(matches, -1))
    unmatched_anchor_indices = tf.squeeze(unmatched_anchor_indices, axis=1)
    # it has shape [num_anchors - num_matches]

    unmatched_reg_targets = tf.zeros([tf.size(unmatched_anchor_indices), 4])
    # it has shape [num_anchors - num_matches, 4]

    matched_anchor_indices = tf.to_int32(matched_anchor_indices)
    unmatched_anchor_indices = tf.to_int32(unmatched_anchor_indices)

    reg_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_reg_targets, unmatched_reg_targets]
    )
    reg_targets.set_shape([num_anchors, 4])
    matched_cls_targets = tf.gather(groundtruth_labels, matched_gt_indices)
    # it has shape [num_matches, num_classes]

    matched_cls_targets = tf.pad(matched_cls_targets, [[0, 0], [1, 0]])
    # it has shape [num_matches, num_classes + 1]

    # one-hot encoding for background class
    unmatched_cls_target = tf.constant([[1.0] + num_classes*[0.0]], tf.float32)

    unmatched_cls_targets = tf.tile(
        unmatched_cls_target,
        tf.stack([tf.size(unmatched_anchor_indices), 1])
    )  # shape [num_anchors - num_matches, num_classes + 1]

    cls_targets = tf.dynamic_stitch(
        [matched_anchor_indices, unmatched_anchor_indices],
        [matched_cls_targets, unmatched_cls_targets]
    )
    cls_targets.set_shape([num_anchors, num_classes + 1])

    return reg_targets, cls_targets
