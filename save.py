import tensorflow as tf
import json

from model import model_fn
tf.logging.set_verbosity('INFO')


params = json.load(open('config.json'))
model_params = params['model_params']

config = tf.ConfigProto()
config.gpu_options.visible_device_list = '0'

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    model_dir=model_params['model_dir'],
    session_config=config
)

estimator = tf.estimator.Estimator(model_fn, params=model_params, config=run_config)


def serving_input_receiver_fn():
    images = tf.placeholder(dtype=tf.float32, shape=[None, 384, 640, 3], name='image_tensor')
    features = {'images': tf.transpose(images*(1.0/255.0), perm=[0, 3, 1, 2])}
    return tf.estimator.export.ServingInputReceiver(features, {'images': images})


estimator.export_savedmodel(
    'export/run00', serving_input_receiver_fn
)
