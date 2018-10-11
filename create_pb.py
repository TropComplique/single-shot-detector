import tensorflow as tf
import os
import shutil
import json
from model import model_fn
from detector.input_pipeline.pipeline import resize_keeping_aspect_ratio
tf.logging.set_verbosity('INFO')


"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""


CONFIG = 'config.json'
OUTPUT_FOLDER = 'export/'  # for savedmodel
GPU_TO_USE = '0'
PB_FILE_PATH = 'inference/model.pb'
MIN_DIMENSION = 640
WIDTH, HEIGHT = None, None
BATCH_SIZE = 1  # must be an integer
assert BATCH_SIZE == 1

params = json.load(open(CONFIG))
params['min_dimension'] = MIN_DIMENSION,


def export_savedmodel():
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE
    run_config = tf.estimator.RunConfig()
    run_config = run_config.replace(
        model_dir=params['model_dir'],
        session_config=config
    )
    estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)

    def serving_input_receiver_fn():
        raw_images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, HEIGHT, WIDTH, 3], name='images')

        images = tf.to_float(raw_images)
        images = tf.squeeze(images, 0)
        resized_images, box_scaler = resize_keeping_aspect_ratio(images, MIN_DIMENSION, divisor=128)

        features = {
            'images': (1.0/255.0) * tf.expand_dims(resized_images, 0),
            'box_scaler': box_scaler
        }
        return tf.estimator.export.ServingInputReceiver(features, {'images': raw_images})

    shutil.rmtree(OUTPUT_FOLDER, ignore_errors=True)
    os.mkdir(OUTPUT_FOLDER)
    estimator.export_savedmodel(OUTPUT_FOLDER, serving_input_receiver_fn)


def convert_to_pb():

    subfolders = os.listdir(OUTPUT_FOLDER)
    assert len(subfolders) == 1
    last_saved_model = os.path.join(OUTPUT_FOLDER, subfolders[0])

    graph = tf.Graph()
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = GPU_TO_USE

    with graph.as_default():
        with tf.Session(graph=graph, config=config) as sess:
            tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], last_saved_model)

            # output ops
            keep_nodes = ['boxes', 'labels', 'scores', 'num_boxes']

            input_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, graph.as_graph_def(),
                output_node_names=keep_nodes
            )

            keep_nodes += [n.name for n in graph.as_graph_def().node if 'nms' in n.name]
            output_graph_def = tf.graph_util.remove_training_nodes(
                input_graph_def, protected_nodes=keep_nodes
            )

            with tf.gfile.GFile(PB_FILE_PATH, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


export_savedmodel()
convert_to_pb()
