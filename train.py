import os
import tensorflow as tf
import json
from model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


"""
This script does the training.
Also it runs the evaluation now and then during the training.
"""

GPU_TO_USE = '0'
CONFIG = 'config.json'  # 'config_mobilenet.json' or 'config_shufflenet.json'
params = json.load(open(CONFIG))


def get_input_fn(is_training=True):

    dataset_path = params['train_dataset'] if is_training else params['val_dataset']
    filenames = os.listdir(dataset_path)
    filenames = [n for n in filenames if n.endswith('.tfrecords')]
    filenames = [os.path.join(dataset_path, n) for n in sorted(filenames)]

    def input_fn():
        with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
            pipeline = Pipeline(filenames, is_training, params)
        return pipeline.dataset

    return input_fn


session_config = tf.ConfigProto(allow_soft_placement=True)
session_config.gpu_options.visible_device_list = GPU_TO_USE
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=params['model_dir'], session_config=session_config,
    save_summary_steps=600, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


if params['backbone'] == 'mobilenet':
    scope_to_restore = 'MobilenetV1/*'
elif params['backbone'] == 'shufflenet':
    scope_to_restore = 'ShuffleNetV2/*'
warm_start = tf.estimator.WarmStartSettings(
    params['pretrained_checkpoint'], [scope_to_restore]
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(
    model_fn, params=params, config=run_config,
    warm_start_from=warm_start
)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 3, throttle_secs=3600 * 3,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
