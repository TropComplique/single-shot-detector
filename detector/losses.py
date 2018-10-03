import tensorflow as tf
from detector.utils import batch_decode


def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return weights * tf.reduce_sum(loss, axis=2)


def focal_loss(predictions, targets, weights, gamma=2.0, alpha=0.25):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, num_anchors].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    positive_label_mask = tf.equal(targets, 1.0)

    negative_log_p_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)
    probabilities = tf.sigmoid(predictions)
    p_t = tf.where(positive_label_mask, probabilities, 1.0 - probabilities)
    # they all have shape [batch_size, num_anchors, num_classes]

    modulating_factor = tf.pow(1.0 - p_t, gamma)
    weighted_loss = tf.where(
        positive_label_mask,
        alpha * negative_log_p_t,
        (1.0 - alpha) * negative_log_p_t
    )
    focal_loss = modulating_factor * weighted_loss
    # they all have shape [batch_size, num_anchors, num_classes]

    return weights * tf.reduce_sum(focal_loss, axis=2)


def usual_classification_loss(predictions, targets, weights):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=targets, logits=predictions
    )
    return weights * tf.reduce_sum(cross_entropy, axis=2)


def apply_hard_mining(
        loc_losses, cls_losses,
        encoded_boxes, matches, anchors,
        loss_to_use='classification',
        loc_loss_weight=1.0, cls_loss_weight=1.0,
        num_hard_examples=3000, nms_threshold=0.99,
        max_negatives_per_positive=3, min_negatives_per_image=0):
    """
    Online hard example mining (OHEM) implementation.

    Arguments:
        loc_losses: a float tensor with shape [batch_size, num_anchors].
        cls_losses: a float tensor with shape [batch_size, num_anchors].
        encoded_boxes: a float tensor with shape [batch_size, num_anchors, 4].
        matches: an int tensor with shape [batch_size, num_anchors].
        anchors: a float tensor with shape [num_anchors, 4].
        loss_to_use: a string, only possible values are ['classification', 'both'].
        loc_loss_weight: a float number.
        cls_loss_weight: a float number.
        num_hard_examples: an integer.
        nms_threshold: a float number.
        max_negatives_per_positive: a float number.
        min_negatives_per_image: an integer.
    Returns:
        two float tensors with shape [].
    """
    decoded_boxes = batch_decode(encoded_boxes, anchors)
    # it has shape [batch_size, num_anchors, 4]

    # all these tensors must have static first dimension (batch size)
    decoded_boxes_list = tf.unstack(decoded_boxes, axis=0)
    loc_losses_list = tf.unstack(loc_losses, axis=0)
    cls_losses_list = tf.unstack(cls_losses, axis=0)
    matches_list = tf.unstack(matches, axis=0)
    # they all lists with length = batch_size

    mined_loc_losses, mined_cls_losses = [], []
    num_positives_list, num_negatives_list = [], []

    # do ohem for each image in the batch
    for i, boxes in enumerate(decoded_boxes_list):

        image_losses = cls_losses_list[i] * cls_loss_weight
        if loss_to_use == 'both':
            image_losses += (loc_losses_list[i] * loc_loss_weight)
        # it has shape [num_anchors]

        selected_indices = tf.image.non_max_suppression(
            boxes, image_losses, num_hard_examples, nms_threshold
        )
        subsampled = subsample_selection_to_desired_neg_pos_ratio(
             selected_indices, matches_list[i],
             max_negatives_per_positive, min_negatives_per_image
        )
        selected_indices, num_positives, num_negatives = subsampled

        mined_loc_losses.append(tf.gather(loc_losses_list[i], selected_indices))
        mined_cls_losses.append(tf.gather(cls_losses_list[i], selected_indices))
        num_positives_list.append(num_positives)
        num_negatives_list.append(num_negatives)

    mean_num_positives = tf.reduce_mean(tf.stack(num_positives_list, axis=0), axis=0)
    mean_num_negatives = tf.reduce_mean(tf.stack(num_negatives_list, axis=0), axis=0)
    tf.summary.scalar('mean_num_positives', mean_num_positives)
    tf.summary.scalar('mean_num_negatives', mean_num_negatives)

    loc_loss = tf.reduce_sum(tf.concat(mined_loc_losses, axis=0), axis=0)
    cls_loss = tf.reduce_sum(tf.concat(mined_cls_losses, axis=0), axis=0)
    return loc_loss, cls_loss


def subsample_selection_to_desired_neg_pos_ratio(
        indices, match, max_negatives_per_positive, min_negatives_per_image):
    """
    Subsample a collection of selected indices
    to a desired negative to positive ratio.

    Arguments:
        indices: an int or long tensor with shape [M],
            it represents a collection of selected anchor indices.
        match: an int tensor with shape [num_anchors].
        max_negatives_per_positive: a float number, maximum number
            of negatives for each positive anchor.
        min_negatives_per_image: an integer, minimum number of negative anchors for a given
            image. Allows sampling negatives in image without any positive anchors.
    Returns:
        selected_indices: an int or long tensor with shape [M'] and with M' <= M.
            It represents a collection of selected anchor indices.
        num_positives: an int tensor with shape []. It represents the
            number of positive examples in selected set of indices.
        num_negatives: an int tensor with shape []. It represents the
            number of negative examples in selected set of indices.
    """
    positives_indicator = tf.gather(tf.greater_equal(match, 0), indices)
    negatives_indicator = tf.logical_not(positives_indicator)
    # they have shape [num_hard_examples]

    # all positives in `indices` will be kept
    num_positives = tf.reduce_sum(tf.to_int32(positives_indicator))
    max_negatives = tf.maximum(
        min_negatives_per_image,
        tf.to_int32(max_negatives_per_positive * tf.to_float(num_positives))
    )

    top_k_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator)),
        max_negatives
    )
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, top_k_negatives_indicator)
    )  # shape [num_hard_examples, 1]
    subsampled_selection_indices = tf.squeeze(subsampled_selection_indices, axis=1)
    selected_indices = tf.gather(indices, subsampled_selection_indices)

    num_negatives = tf.size(subsampled_selection_indices) - num_positives
    return selected_indices, num_positives, num_negatives
