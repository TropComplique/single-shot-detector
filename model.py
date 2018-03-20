import os
import re
import tensorflow as tf

from src import SSD, AnchorGenerator, FeatureExtractor
from src.backbones import mobilenet_v1_base
from evaluation_utils import Evaluator


def model_fn(features, labels, mode, params, config):
    """This is a function for creating a computational tensorflow graph.
    The function is in format required by tf.estimator.
    """

    # the base network
    def backbone(images, is_training):
        return mobilenet_v1_base(
            images, is_training,
            depth_multiplier=params['depth_multiplier']
        )

    # ssd anchor maker
    anchor_generator = AnchorGenerator(
        scales=params['scales'],
        min_scale=params['min_scale'],
        max_scale=params['max_scale'],
        aspect_ratios=params['aspect_ratios'],
        interpolated_scale_aspect_ratio=params['interpolated_scale_aspect_ratio'],
        reduce_boxes_in_lowest_layer=params['reduce_boxes_in_lowest_layer']
    )

    # add additional layers to the base network
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    feature_extractor = FeatureExtractor(backbone, is_training)

    # add box/label predictors to the feature extractor
    ssd = SSD(features['images'], feature_extractor, anchor_generator, params['num_classes'])

    # use a pretrained backbone network
    if params['pretrained_checkpoint'] is not None:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(params['pretrained_checkpoint'], {'MobilenetV1/': 'MobilenetV1/'})

    # add NMS to the graph
    if not is_training:
        predictions = ssd.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        # rename prediction keys
        name_map = {
            'boxes': 'detection_boxes', 'scores': 'detection_scores',
            'labels': 'detection_classes', 'num_boxes': 'num_detections'
        }
        # this is required for exporting a savedmodel
        export_outputs = tf.estimator.export.PredictOutput({
            name: tf.identity(tensor, name_map[name])
            for name, tensor in predictions.items()
        })
        return tf.estimator.EstimatorSpec(
            mode, predictions=predictions,
            export_outputs={'outputs': export_outputs}
        )

    # add L2 regularization
    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = ssd.loss(labels, params)
    tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
    tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('localization_loss', losses['localization_loss'])
    tf.summary.scalar('classification_loss', losses['classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    if mode == tf.estimator.ModeKeys.EVAL:

        filenames = features['filenames']
        batch_size = filenames.shape.as_list()[0]
        assert batch_size == 1

        with tf.name_scope('evaluator'):
            evaluator = Evaluator(num_classes=params['num_classes'])
            eval_metric_ops = evaluator.get_metric_ops(filenames, labels, predictions)

        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss,
            eval_metric_ops=eval_metric_ops
        )

    assert mode == tf.estimator.ModeKeys.TRAIN
    with tf.variable_scope('learning_rate'):
        global_step = tf.train.get_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # you can freeze some variables
        trainable_var = tf.trainable_variables()
        regexp = re.compile(params['freeze'])
        var_list = [v for v in trainable_var if not bool(regexp.search(v.name))]

        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=var_list)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step)

    for g, v in grads_and_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)
        tf.summary.histogram(v.name[:-2] + '_grad_hist', g)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)


class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""
    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )
    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name and 'depthwise_weights' not in v.name]
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
