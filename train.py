import tensorflow as tf
import json

from model import model_fn, IteratorInitializerHook
from src.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


params = json.load(open('config.json'))
model_params = params['model_params']
input_params = params['input_pipeline_params']


def get_input_fn(is_training=True):

    image_size = input_params['image_size']
    iterator_initializer_hook = IteratorInitializerHook()
    filename = input_params['train_dataset'] if is_training else input_params['val_dataset']
    batch_size = input_params['batch_size'] if is_training else 1
    # for evaluation it's important to set batch_size to 1

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(
                filename,
                batch_size=batch_size, image_size=image_size,
                repeat=is_training, shuffle=is_training,
                augmentation=is_training
            )
            features, labels = pipeline.get_batch()
        iterator_initializer_hook.iterator_initializer_func = lambda sess: sess.run(pipeline.init)
        return features, labels

    return input_fn, iterator_initializer_hook


config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config,
    save_summary_steps=100,
    save_checkpoints_secs=300,
    log_step_count_steps=25
)

train_input_fn, train_iterator_initializer_hook = get_input_fn(is_training=True)
val_input_fn, val_iterator_initializer_hook = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


train_spec = tf.estimator.TrainSpec(
    train_input_fn, max_steps=input_params['num_steps'],
    hooks=[train_iterator_initializer_hook]
)
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None,
    hooks=[val_iterator_initializer_hook],
    start_delay_secs=300, throttle_secs=300
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
