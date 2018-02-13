import tensorflow as tf
from evaluation_utils import Evaluator


class Metric:
    def __init__(self, num_classes):
        self.evaluator = Evaluator(num_classes)

    def get_estimator_eval_metric_ops(
        self, image_id, groundtruth_boxes,
        groundtruth_classes, detection_boxes,
        detection_scores, detection_classes):
        """Returns a dictionary of eval metric ops to use with `tf.EstimatorSpec`.
        Note that once value_op is called, the detections and groundtruth added via
        update_op are cleared.
        Args:
          image_id: Unique string/integer identifier for the image.
          groundtruth_boxes: float32 tensor of shape [num_boxes, 4] containing
            `num_boxes` groundtruth boxes of the format
            [ymin, xmin, ymax, xmax] in absolute image coordinates.
          groundtruth_classes: int32 tensor of shape [num_boxes] containing
            1-indexed groundtruth classes for the boxes.
          detection_boxes: float32 tensor of shape [num_boxes, 4] containing
            `num_boxes` detection boxes of the format [ymin, xmin, ymax, xmax]
            in absolute image coordinates.
          detection_scores: float32 tensor of shape [num_boxes] containing
            detection scores for the boxes.
          detection_classes: int32 tensor of shape [num_boxes] containing
            1-indexed detection classes for the boxes.
        Returns:
          a dictionary of metric names to tuple of value_op and update_op that can
          be used as eval metric ops in tf.EstimatorSpec. Note that all update ops
          must be run together and similarly all value ops must be run together to
          guarantee correct behaviour.
        """
        def update_op(
                image_id,
                groundtruth_boxes,
                groundtruth_classes,
                detection_boxes,
                detection_scores,
                detection_classes):
            self.evaluator.add_groundtruth(images, boxes, labels)
            self.evaluator.add_detections(images, boxes, labels, scores, num_boxes)

    update_op = tf.py_func(update_op, [image_id,
                                       groundtruth_boxes,
                                       groundtruth_classes,
                                       detection_boxes,
                                       detection_scores,
                                       detection_classes], [], stateful=True)
    metric_names = ['DetectionBoxes_Precision/mAP',
                    'DetectionBoxes_Precision/mAP@.50IOU',
                    'DetectionBoxes_Precision/mAP@.75IOU',
                    'DetectionBoxes_Precision/mAP (large)',
                    'DetectionBoxes_Precision/mAP (medium)',
                    'DetectionBoxes_Precision/mAP (small)',
                    'DetectionBoxes_Recall/AR@1',
                    'DetectionBoxes_Recall/AR@10',
                    'DetectionBoxes_Recall/AR@100',
                    'DetectionBoxes_Recall/AR@100 (large)',
                    'DetectionBoxes_Recall/AR@100 (medium)',
                    'DetectionBoxes_Recall/AR@100 (small)']
    if self._include_metrics_per_category:
      for category_dict in self._categories:
        metric_names.append('DetectionBoxes_PerformanceByCategory/mAP/' +
                            category_dict['name'])

    def first_value_func():
      self._metrics = self.evaluate()
      self.clear()
      return np.float32(self._metrics[metric_names[0]])

    def value_func_factory(metric_name):
      def value_func():
        return np.float32(self._metrics[metric_name])
      return value_func

    first_value_op = tf.py_func(first_value_func, [], tf.float32)
    eval_metric_ops = {metric_names[0]: (first_value_op, update_op)}
    with tf.control_dependencies([first_value_op]):
      for metric_name in metric_names[1:]:
        eval_metric_ops[metric_name] = (tf.py_func(
            value_func_factory(metric_name), [], np.float32), update_op)
    return eval_metric_ops
