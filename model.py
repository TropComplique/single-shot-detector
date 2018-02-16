import os
import numpy as np
import tensorflow as tf

from src import SSD, AnchorGenerator, FeatureExtractor
from src.backbones import mobilenet_v1_base
from evaluation_utils import Evaluator


def model_fn(features, labels, mode, params, config):

    # the base network
    def backbone(images, is_training):
        return mobilenet_v1_base(
            images, is_training, depth_multiplier=params['depth_multiplier']
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
    
    # use a pretrained backbone network
    if params['pretrained_checkpoint'] is not None:
        with tf.name_scope('init_from_checkpoint'):
            tf.train.init_from_checkpoint(params['pretrained_checkpoint'], {'MobilenetV1/': 'MobilenetV1/'})

    if not is_training:
        predictions = ssd.get_predictions(
            score_threshold=params['score_threshold'],
            iou_threshold=params['iou_threshold'],
            max_boxes_per_class=params['max_boxes_per_class']
        )

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    with tf.name_scope('weight_decay'):
        add_weight_decay(params['weight_decay'])
        regularization_loss = tf.losses.get_regularization_loss()

    # create localization and classification losses
    losses = ssd.loss(labels, params)
    tf.losses.add_loss(params['localization_loss_weight'] * losses['localization_loss'])
    tf.losses.add_loss(params['classification_loss_weight'] * losses['classification_loss'])
    total_loss = tf.losses.get_total_loss(add_regularization_losses=True)

    tf.summary.scalar('regularization_loss', regularization_loss)
    tf.summary.scalar('localization_loss', losses['localization_loss'])
    tf.summary.scalar('classification_loss', losses['classification_loss'])

    if mode == tf.estimator.ModeKeys.EVAL:
        evaluator = Evaluator(params['num_classes'])
        with tf.name_scope('image_drawer'):
            image_summary = get_image_visualizer(features['images'], predictions)
            summary_hook = tf.train.SummarySaverHook(
                save_steps=10,
                output_dir=os.path.join(config.model_dir, 'eval'),
                summary_op=image_summary
            )
        with tf.name_scope('evaluator'):
            eval_metric_ops = evaluator.get_metric_ops(features['filenames'], labels, predictions)
        return tf.estimator.EstimatorSpec(
            mode, loss=total_loss, eval_metric_ops=eval_metric_ops,
            evaluation_hooks=[summary_hook]
        )

    assert mode == tf.estimator.ModeKeys.TRAIN

    global_step = tf.train.get_global_step()
    with tf.variable_scope('learning_rate'):
        learning_rate = tf.train.piecewise_constant(global_step, params['lr_boundaries'], params['lr_values'])
        tf.summary.scalar('learning_rate', learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate, decay=0.9, momentum=0.9, epsilon=1.0
        )
        grads_and_vars = optimizer.compute_gradients(total_loss, var_list=None)
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
    kernels = [v for v in trainable_vars if '/weights' in v.name]

    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)

from PIL import Image, ImageDraw, ImageFont
def get_image_visualizer(images, predictions, max_outputs=20):
    images = tf.transpose(images, perm=[0, 2, 3, 1])  # to 'NHWC' format
    tensors = [
        images[:max_outputs], 
        predictions['boxes'][:max_outputs], 
        predictions['scores'][:max_outputs],
        predictions['labels'][:max_outputs], 
        predictions['num_boxes'][:max_outputs]
    ]
    def draw_boxes(images, boxes, scores, labels, num_boxes):
        result = []
        for i, b, s, n in zip(images, boxes, scores, num_boxes):
            result.append(draw_boxes_on_image(i, b[:n], s[:n]))
        return np.stack(result, axis=0)
    images_with_boxes = tf.py_func(draw_boxes, tensors, tf.uint8, stateful=False)
    return tf.summary.image('predictions', images_with_boxes, max_outputs)
    
    
    
def draw_boxes_on_image(image, boxes, scores):
    image = (255.0*image).astype('uint8')
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    scale = np.array([height, width, height, width], dtype='float')
    boxes = boxes*scale
    
    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        
        text = str(score)
        fill = (255, 255, 255, 45)
        outline = 'red'

        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.rectangle(
            [(xmin, ymin), (xmin + 4*len(text) + 4, ymin + 10)],
            fill='white', outline='white'
        )
        #font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu-font-family/Ubuntu-B.ttf', size=8)
        draw.text((xmin + 1, ymin + 1), text, fill='red')

    return np.array(image)

    
    
    
    
    
    
    
    
    
    