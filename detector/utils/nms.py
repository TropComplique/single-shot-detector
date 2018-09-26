import tensorflow as tf
from detector.constants import PARALLEL_ITERATIONS


def multiclass_non_max_suppression(
        boxes, scores,
        score_threshold, iou_threshold,
        max_boxes_per_class):
    """Multi-class version of non maximum suppression. It operates independently
    for each class. Also it prunes boxes with score less than a provided
    threshold prior to applying NMS.

    Arguments:
        boxes: a float tensor with shape [N, num_classes, 4],
            normalized to [0, 1] range.
        scores: a float tensor with shape [N, num_classes].
        score_thresh: a float number.
        iou_threshold: a float number, threshold for IoU.
        max_boxes_per_class: an integer, maximum number of retained boxes per class.
    Returns:
        selected_boxes: a float tensor with shape [N', 4].
        selected_scores: a float tensor with shape [N'].
        selected_classes: an int tensor with shape [N'].

        where 0 <= N' <= num_classes * max_boxes_per_class.
    """
    boxes_list = tf.unstack(boxes, axis=1)
    scores_list = tf.unstack(scores, axis=1)

    selected_boxes, selected_scores, selected_classes = [], [], []
    for i in range(len(boxes_list)):

        selected_indices = tf.image.non_max_suppression(
            boxes_list[i], scores_list[i], max_output_size=max_boxes_per_class,
            iou_threshold=iou_threshold, score_threshold=score_threshold
        )

        selected_boxes += [tf.gather(boxes_list[i], selected_indices)]
        selected_scores += [tf.gather(scores_list[i], selected_indices)]
        selected_classes += [i * tf.ones_like(selected_indices)]

    selected_boxes = tf.concat(selected_boxes, axis=0)
    selected_scores = tf.concat(selected_scores, axis=0)
    selected_classes = tf.to_int32(tf.concat(selected_classes, axis=0))
    return selected_boxes, selected_scores, selected_classes


def batch_multiclass_non_max_suppression(
        boxes, scores, score_threshold,
        iou_threshold, max_boxes_per_class):
    """Same as multiclass_non_max_suppression but for a batch of images.

    Arguments:
        boxes: a float tensor with shape [batch_size, N, num_classes, 4].
        scores: a float tensor with shape [batch_size, N, num_classes].
    Returns:
        boxes: a float tensor with shape [batch_size, N', 4].
        scores: a float tensor with shape [batch_size, N'].
        classes: an int tensor with shape [batch_size, N'].
        num_detections: an int tensor with shape [batch_size].

        N' = max_boxes_per_class * num_classes
    """
    def fn(x):
        boxes, scores = x
        boxes, scores, classes = multiclass_non_max_suppression(
            boxes, scores, score_threshold,
            iou_threshold, max_boxes_per_class
        )
        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        max_detections = max_boxes_per_class * num_classes

        zero_padding = max_detections - num_boxes
        boxes = tf.pad(boxes, [[0, zero_padding], [0, 0]])
        scores = tf.pad(scores, [[0, zero_padding]])
        classes = tf.pad(classes, [[0, zero_padding]])

        boxes.set_shape([max_detections, 4])
        scores.set_shape([max_detections])
        classes.set_shape([max_detections])
        return boxes, scores, classes, num_boxes

    boxes, scores, classes, num_detections = tf.map_fn(
        fn, [boxes, scores],
        dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
        parallel_iterations=PARALLEL_ITERATIONS,
        back_prop=False, swap_memory=False, infer_shape=True
    )
    return boxes, scores, classes, num_detections
