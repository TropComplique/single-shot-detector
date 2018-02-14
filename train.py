import tensorflow as tf

from estimator import model_fn
from utils import IteratorInitializerHook
from src.input_pipeline import Pipeline


iterator_initializer_hook = IteratorInitializerHook()

def train_input_fn():
    pipeline = Pipeline(
        'data/val.tfrecords',
        batch_size=24, image_size=(640, 360),
        repeat=True, shuffle=False, augmentation=False
    )
    features, labels = pipeline.get_batch()
    iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(pipeline.init)
    return features, labels

params = {
    'depth_multiplier': 1.0,
    'min_scale': 0.1, 'max_scale': 0.9,
    'aspect_ratios': (1.0, 0.5, 0.3),
    'interpolated_scale_aspect_ratio': 1.0,
    'reduce_boxes_in_lowest_layer': True,
    'num_classes': 1,
    'score_threshold': 0.1, 'iou_threshold': 0.6, 'max_boxes_per_class': 20,
    'weight_decay': 1e-5,
    'localization_loss_weight': 1.0, 'classification_loss_weight': 1.0,
    'initial_lr': 1e-4,
    'lr_reduce_factor': 0.1,
    'loc_loss_weight': 1.0,
    'cls_loss_weight': 1.0,
    'num_hard_examples': 3000, 
    'nms_threshold': 0.99,
    'max_negatives_per_positive': 3.0, 
    'min_negatives_per_image': 0
}   
    
estimator = tf.estimator.Estimator(model_fn, model_dir='model', params=params)
estimator.train(train_input_fn, hooks=[iterator_initializer_hook], max_steps=1000)
# estimator.evaluate(input_fn, hooks=[iterator_initializer_hook], steps=10)  