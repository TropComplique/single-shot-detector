import os
import re
import argparse
import configparser
import importlib
import shutil
import tensorflow as tf
import tensorflow.contrib.slim as slim

from src import SSD, AnchorGenerator, FeatureExtractor
from src.input_pipeline import Pipeline
from src.backbone import mobilenet_v1_base

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def model_fn(features, labels, mode, params):

    def backbone(images, is_training):
        return mobilenet_v1_base(images, is_training, min_depth=8, depth_multiplier=1.0)

    anchor_generator = AnchorGenerator(
        min_scale=0.2, max_scale=0.95,
        aspect_ratios=(1.0, 2.0, 3.0, 0.5, 0.333),
        interpolated_scale_aspect_ratio=1.0,
        reduce_boxes_in_lowest_layer=True
    )

    is_training = False if tf.estimator.ModeKeys.PREDICT or mode == tf.estimator.ModeKeys.EVAL else True
    feature_extractor = FeatureExtractor(backbone, is_training)
    ssd = SSD(features['images'], feature_extractor, anchor_generator, num_classes)

    if mode == tf.estimator.ModeKeys.PREDICT:
        boxes, scores, classes, num_detections = ssd.get_predictions(
            score_threshold=0.1, iou_threshold=0.6,
            max_boxes_per_class=20
        )
        predictions = {
            'boxes': boxes, 'scores': scores,
            'classes': classes, 'num_detections': num_detections
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = ssd.loss(labels['boxes'], labels['labels'], labels['num_boxes'])

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
