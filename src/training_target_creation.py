import tensorflow as tf
from .utils import encode, iou


def get_targets(anchors, groundtruth_boxes, groundtruth_labels, num_classes, threshold=0.5):
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
        matches: an int tensor with shape [num_anchors].
    """
    with tf.name_scope('matching'):
        N = tf.shape(groundtruth_boxes)[0]
        num_anchors = anchors.shape[0]
        matches = tf.cond(
            tf.greater(N, 0),
            lambda: _match(anchors, groundtruth_boxes, threshold), 
            lambda: -1 * tf.ones([num_anchors], dtype=tf.int32)
        )

    with tf.name_scope('training_target_creation'):
        reg_targets, cls_targets = _create_targets(
            anchors, groundtruth_boxes, groundtruth_labels,
            matches, num_classes
        )

    return cls_targets, reg_targets, matches


def _match(anchors, groundtruth_boxes, threshold=0.5):
    """
    Arguments:
        anchors: a float tensor with shape [num_anchors, 4].
        groundtruth_boxes: a float tensor with shape [N, 4].
        threshold: a float number.
    Returns:
        an int tensor with shape [num_anchors].
    """
    num_anchors = anchors.shape[0]

    # for each anchor box choose the groundtruth box with largest iou
    similarity_matrix = iou(groundtruth_boxes, anchors)  # shape [N, num_anchors]
    matches = tf.to_int32(tf.argmax(similarity_matrix, axis=0))  # shape [num_anchors]
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
    forced_matches_ids = tf.to_int32(tf.argmax(similarity_matrix, 1))  # shape [N]

    col_range = tf.range(tf.shape(similarity_matrix)[1], dtype=tf.int32)
    keep_matches_ids, _ = tf.setdiff1d(col_range, forced_matches_ids)
    # note: `col_range` equals to disjoint union of `forced_matches_ids` and `keep_matches_ids`,
    # `keep_matches_ids` - anchor boxes that have no forced groundtruth boxes

    forced_matches_values = tf.range(tf.shape(similarity_matrix)[0], dtype=tf.int32)
    keep_matches_values = tf.gather(matches, keep_matches_ids)

    # set matches[forced_matches_ids] = [0, 1, 2, ..., N].
    matches = tf.dynamic_stitch(
        [forced_matches_ids, keep_matches_ids],
        [forced_matches_values, keep_matches_values]
    )
    # in other words:
    # matches[forced_matches_ids[i]] = forced_matches_values[i]
    # matches[keep_matches_ids[i]] = keep_matches_values[i]

    # so, in the end, we matched each row to at least one column
    matches.set_shape(num_anchors)
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
    num_anchors = anchors.shape[0]

    matched_anchor_indices = tf.where(tf.greater_equal(matches, 0))  # shape [num_matches, 1]
    matched_anchor_indices = tf.squeeze(matched_anchor_indices, axis=1)
    #print(matched_anchor_indices)
    #matched_anchor_indices.set_shape([None])
    matched_gt_indices = tf.gather(matches, matched_anchor_indices)  # shape [num_matches]

    matched_anchors = tf.gather(anchors, matched_anchor_indices)  # shape [num_matches, 4]
    matched_gt_boxes = tf.gather(groundtruth_boxes, matched_gt_indices)  # shape [num_matches, 4]
    #matched_gt_boxes = tf.Print(matched_gt_boxes, [matched_gt_boxes, matched_anchors])
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

    matched_cls_targets = tf.pad(matched_cls_targets, tf.constant([[0, 0], [1, 0]]))
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
