import tensorflow as tf

from src import SSD, AnchorGenerator, FeatureExtractor
from src.input_pipeline import Pipeline
from src.backbones import mobilenet_v1_base
from metric_ops import Metric


def model_fn(features, labels, mode, params):
    
    # the base network
    def backbone(images, is_training):
        return mobilenet_v1_base(images, is_training, min_depth=8, depth_multiplier=1.0)

    anchor_generator = AnchorGenerator(
        min_scale=0.2, max_scale=0.95,
        aspect_ratios=(1.0, 2.0, 3.0, 0.5, 0.333),
        interpolated_scale_aspect_ratio=1.0,
        reduce_boxes_in_lowest_layer=True
    )

    is_training = mode == tf.estimator.ModeKeys.TRAIN
    feature_extractor = FeatureExtractor(backbone, is_training)
    ssd = SSD(features['images'], feature_extractor, anchor_generator, num_classes=1)
    
    boxes, scores, classes, num_detections = ssd.get_predictions(
        score_threshold=0.1, iou_threshold=0.6,
        max_boxes_per_class=20
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'boxes': boxes, 'scores': scores,
            'classes': classes, 'num_detections': num_detections
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = ssd.loss(labels['boxes'], labels['labels'], labels['num_boxes'])
    
    metrics = Metric(1)
    eval_metric_ops = metrics.get_estimator_eval_metric_ops(
        features['filenames'], labels['boxes'],
        labels['labels'], labels['num_boxes'], 
        boxes, classes, scores, num_detections
    )

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=eval_metric_ops)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.001)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.variable_scope('optimizer'):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

iterator_initializer_hook = IteratorInitializerHook()

def input_fn():
    pipeline = Pipeline(
        'data/val.tfrecords',
        batch_size=2, image_size=(640, 360),
        repeat=True, shuffle=False, augmentation=True
    )
    batch = pipeline.get_batch()
    features = {'images': batch['images'], 'filenames': batch['filenames']}
    labels = {'boxes': batch['boxes'], 'labels': batch['labels'], 'num_boxes': batch['num_boxes']}
    iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(pipeline.init)
    return features, labels




estimator = tf.estimator.Estimator(model_fn, model_dir='model', params={})
estimator.train(input_fn, hooks=[iterator_initializer_hook], max_steps=1000)
#estimator.evaluate(input_fn, hooks=[iterator_initializer_hook], steps=10)  

