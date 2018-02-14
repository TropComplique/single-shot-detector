import tensorflow as tf

from src import SSD, AnchorGenerator, FeatureExtractor
from src.backbones import mobilenet_v1_base
from evaluation_utils import Evaluator
from utils import add_histograms, add_weight_decay


def model_fn(features, labels, mode, params):
    
    # the base network
    def backbone(images, is_training):
        return mobilenet_v1_base(
            images, is_training, 
            depth_multiplier=params['depth_multiplier']
        )
    
    # ssd anchor maker
    anchor_generator = AnchorGenerator(
        min_scale=params['min_scale'], max_scale=params['max_scale'],
        aspect_ratios=params['aspect_ratios'],
        interpolated_scale_aspect_ratio=params['interpolated_scale_aspect_ratio'],
        reduce_boxes_in_lowest_layer=params['reduce_boxes_in_lowest_layer']
    )

    # add additional layers to the base network
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    feature_extractor = FeatureExtractor(backbone, is_training)
    
    # add box/label predictors to the feature extractor
    ssd = SSD(features['images'], feature_extractor, anchor_generator, params['num_classes'])
    
    if not is_training:
        predictions = ssd.get_predictions(
            score_threshold=params['score_threshold'], 
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    with tf.variable_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
       
    # create localization and classification losses
    with tf.name_scope('loss'):
        losses = ssd.loss(labels, params)
        tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
        tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
        regularization_loss = tf.losses.get_regularization_loss()
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        
        tf.summary.scalar('regularization_loss', regularization_loss)
        tf.summary.scalar('localization_loss', losses['localization_loss'])
        tf.summary.scalar('classification_loss', losses['classification_loss'])

    if mode == tf.estimator.ModeKeys.EVAL:
        evaluator = Evaluator(params['num_classes'])
        eval_metric_ops = evaluator.get_metric_ops(features['filenames'], labels, predictions)
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=eval_metric_ops)
    
    assert mode == tf.estimator.ModeKeys.TRAIN
    
    add_histograms()

    with tf.variable_scope('learning_rate'):
        learning_rate = tf.Variable(
            params['initial_lr'], trainable=False,
            dtype=tf.float32, name='lr'
        )
        # you can reduce learning rate by some factor, usually 0.1
        drop_learning_rate = tf.assign(
            learning_rate, params['lr_reduce_factor']*learning_rate
        )

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.9, momentum=0.9,
        )
        grads_and_vars = optimizer.compute_gradients(total_loss)
        train_op = optimizer.apply_gradients(grads_and_vars)

    return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)

