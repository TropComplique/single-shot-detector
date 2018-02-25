import tensorflow as tf
import json
import argparse
from model import model_fn


"""The purpose of this script is to export a savedmodel."""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o', '--output', type=str, default='export/run00'
    )
    return parser.parse_args()


tf.logging.set_verbosity('INFO')
ARGS = make_args()
params = json.load(open('config.json'))
model_params = params['model_params']
input_pipeline_params = params['input_pipeline_params']
width, height = input_pipeline_params['image_size']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'
run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config
)
estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.float32, shape=[None, height, width, 3], name='image_tensor')
    features = {'images': tf.transpose(images*(1.0/255.0), perm=[0, 3, 1, 2])}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    ARGS.output, serving_input_receiver_fn
)
