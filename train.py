import os
import tensorflow as tf
import json

from model import model_fn, RestoreMovingAverageHook
from detector.input_pipeline import Pipeline
tf.logging.set_verbosity('INFO')


GPU_TO_USE = '1'

# CONFIG = 'config.json'
# params = json.load(open(CONFIG))

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

    "min_dimension": 768,
    "batch_size": 14,
    "image_height": 640,
    "image_width": 640,

    "num_steps": 60000,
}


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
    save_summary_steps=1000, save_checkpoints_secs=1800,
    log_step_count_steps=1000
)


train_input_fn = get_input_fn(is_training=True)
val_input_fn = get_input_fn(is_training=False)
estimator = tf.estimator.Estimator(model_fn, params=params, config=run_config)


train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=params['num_steps'])
eval_spec = tf.estimator.EvalSpec(
    val_input_fn, steps=None, start_delay_secs=3600 * 3, throttle_secs=3600 * 3,
    hooks=[RestoreMovingAverageHook(params['model_dir'])]
)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
