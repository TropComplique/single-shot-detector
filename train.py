import tensorflow as tf
import json

from estimator import model_fn, IteratorInitializerHook
from src.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


params = json.load(open('config.json'))
model_params = params['model_params']
input_pipeline_params = params['input_pipeline_params']


def get_input_fn(is_training=True):

    image_size = input_pipeline_params['image_size']
    iterator_initializer_hook = IteratorInitializerHook()
    filename = 'data/train.tfrecords' if is_training else 'data/val.tfrecords'
    batch_size = input_pipeline_params['batch_size'] if is_training else 1
    augmentation = input_pipeline_params if is_training else None

    def input_fn():
        pipeline = Pipeline(
            filename,
            batch_size=batch_size, image_size=image_size,
            repeat=is_training, shuffle=is_training,
            augmentation=augmentation
        )
        with tf.device('/cpu:0'):
            features, labels = pipeline.get_batch()
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(pipeline.init)
        return features, labels

    return input_fn, iterator_initializer_hook


config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir='models/run00',
    session_config=config,
    save_summary_steps=100,
    save_checkpoints_secs=300
)

train_input_fn, train_iterator_initializer_hook = get_input_fn(is_training=True)
val_input_fn, val_iterator_initializer_hook = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


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
