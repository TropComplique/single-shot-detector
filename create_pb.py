import tensorflow as tf
import os
import shutil
from model import model_fn
from detector.input_pipeline.pipeline import resize_keeping_aspect_ratio


"""
The purpose of this script is to export
the inference graph as a SavedModel.

Also it creates a .pb frozen inference graph.
"""


OUTPUT_FOLDER = 'export/'  # for savedmodel
GPU_TO_USE = '0'
PB_FILE_PATH = 'model.pb'
MIN_DIMENSION = 768
WIDTH, HEIGHT = None, None
BATCH_SIZE = 1  # must be an integer
assert BATCH_SIZE == 1

params = {
    "model_dir": "models/run00",
    "train_dataset": "/home/dan/datasets/COCO/coco_person/train_shards/",
    "val_dataset": "/home/dan/datasets/COCO/coco_person/val_shards/",

    "backbone": "shufflenet",  # 'mobilenet' or 'shufflenet'
    "depth_multiplier": 1.0,
    "weight_decay": 1e-5,
    "num_classes": 1,

    "pretrained_checkpoint": "pretrained/shufflenet_v2_1.0x/model.ckpt-1661328",

    "score_threshold": 0.05, "iou_threshold": 0.5, "max_boxes_per_class": 20,
    "localization_loss_weight": 1.0, "classification_loss_weight": 1.0,

    "use_focal_loss": True,
    "gamma": 2.0,
    "alpha": 0.25,

    "use_ohem": False,
    # "loss_to_use": "classification",
    # "loc_loss_weight": 0.0, "cls_loss_weight": 1.0,
    # "num_hard_examples": 3000, "nms_threshold": 0.99,
    # "max_negatives_per_positive": 3.0, "min_negatives_per_image": 3,

    "lr_boundaries": [40000, 50000],
    "lr_values": [0.01, 0.001, 0.00001],

    "min_dimension": MIN_DIMENSION,
    "batch_size": 14,
    "image_height": 640,
    "image_width": 640,

    "num_steps": 60000,
}

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
        raw_images = tf.placeholder(dtype=tf.uint8, shape=[BATCH_SIZE, None, None, 3], name='images')
        w, h = tf.shape(raw_images)[2], tf.shape(raw_images)[1]

        #with tf.device('/cpu:0'):

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
            #output_graph_def = input_graph_def

            with tf.gfile.GFile(PB_FILE_PATH, 'wb') as f:
                f.write(output_graph_def.SerializeToString())
            print('%d ops in the final graph.' % len(output_graph_def.node))


tf.logging.set_verbosity('INFO')
export_savedmodel()
convert_to_pb()
