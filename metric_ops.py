import tensorflow as tf
from evaluation_utils import Evaluator
import numpy as np



class Metric:
    def __init__(self, num_classes):
        self.evaluator = Evaluator(num_classes)

    def get_estimator_eval_metric_ops(
            self, images, gboxes, glabels, gnum_boxes,
            boxes, labels, scores, num_boxes):

        def update_op(images, gboxes, glabels, gnum_boxes,
                      boxes, labels, scores, num_boxes):
            self.evaluator.add_groundtruth(images, gboxes, glabels, gnum_boxes)
            self.evaluator.add_detections(images, boxes, labels, scores, num_boxes)

        update_op = tf.py_func(
            update_op, 
            [images, gboxes, glabels, gnum_boxes, boxes, labels, scores, num_boxes], 
            [], stateful=True
        )
    
        def value_func():
            result = self.evaluator.evaluate()
            return np.float32(result[0])

        value_op = tf.py_func(value_func, [], [tf.float32])
        eval_metric_ops = {'ap': (value_op, update_op)}

        return eval_metric_ops
