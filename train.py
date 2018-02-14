import tensorflow as tf

from estimator import model_fn, IteratorInitializerHook
from src.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


def get_input_fn(is_training=True):

    iterator_initializer_hook = IteratorInitializerHook()
    filename = 'data/train.tfrecords' if is_training else 'data/val.tfrecords'
    batch_size = 24 if is_training else 10

    def input_fn():
        pipeline = Pipeline(
            filename,
            batch_size=batch_size, image_size=(640, 360),
            repeat=is_training, shuffle=is_training,
            augmentation=is_training
        )
        features, labels = pipeline.get_batch()
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(pipeline.init)
        return features, labels

    return input_fn, iterator_initializer_hook


params = {
    'depth_multiplier': 1.0,
    'num_classes': 1, 'weight_decay': 1e-5,

    # for anchor generator
    'min_scale': 0.05, 'max_scale': 0.8,
    'aspect_ratios': (1.0, 0.6, 0.4, 0.3333, 0.2),
    'interpolated_scale_aspect_ratio': 1.0,
    'reduce_boxes_in_lowest_layer': False,

    # for final NMS
    'score_threshold': 0.1, 'iou_threshold': 0.6, 'max_boxes_per_class': 40.01, 0.005, 0.001, 0.00050,

    # for final loss
    'localization_loss_weight': 1.0, 'classification_loss_weight': 1.0,

    # for OHEM
    'loc_loss_weight': 0.0, 'cls_loss_weight': 1.0,
    'num_hard_examples': 3000, 'nms_threshold': 0.99,
    'max_negatives_per_positive': 3.0, 'min_negatives_per_image': 0

    # for tf.train.piecewise_constant
    'lr_boundaries': [1500, 8000, 15000], 'lr_values': [0.01, 0.005, 0.001, 0.0005]
}

train_input_fn, train_iterator_initializer_hook = get_input_fn(is_training=True)
val_input_fn, val_iterator_initializer_hook = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, model_dir='model', params=params)

train_spec = tf.estimator.TrainSpec(
    train_input_fn, max_steps=10000,
    hooks=[train_iterator_initializer_hook]
)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None,
    hooks=[val_iterator_initializer_hook],
    start_delay_secs=120, throttle_secs=240
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
