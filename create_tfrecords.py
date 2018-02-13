import io
import os
import PIL.Image
import tensorflow as tf
import json
import argparse
from tqdm import tqdm
import sys


"""
The purpose of this script is to create a .tfrecords file from a folder of images and
a folder of annotations. Annotations are in the json format.

Example of use:
python create_tfrecords.py \
    --image_dir=/home/gpu1/dataset/dataai/BBA_000/images_val/ \
    --annotations_dir=/home/gpu1/dataset/dataai/BBA_000/annotations_val/ \
    --output_path=data/val.tfrecords \
    --label_map_path=data/datai_label_map.pbtxt
"""


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--image_dir', type=str, default='data/images'
    )
    parser.add_argument(
        '-a', '--annotations_dir', type=str, default='data/annotations'
    )
    parser.add_argument(
        '-o', '--output', type=str, default='data/data.tfrecords'
    )
    parser.add_argument(
        '-l', '--labels', type=str, default='data/labels.txt'
    )
    return parser.parse_args()


def dict_to_tf_example(annotation, image_dir, labels):
    """Convert dict to tf.Example proto.

    Notice that this function normalizes the bounding
    box coordinates provided by the raw data.

    Arguments:
        data: a dict.
        image_dir: a string, path to the image directory.
        labels: a dict, class name -> unique integer.
    Returns:
        an instance of tf.Example.
    """
    image_path = os.path.join(image_dir, annotation['filename'])
    with tf.gfile.GFile(image_path, 'rb') as f:
        encoded_jpg = f.read()

    # check image format
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPEG!')

    width = int(annotation['size']['width'])
    height = int(annotation['size']['height'])
    assert width > 0 and height > 0
    ymin, xmin, ymax, xmax, classes = [], [], [], [], []

    annotation_name = annotation['filename'][:-4] + '.json'
    if len(annotation['object']) == 0:
        print(annotation_name, 'is without any objects!')

    for obj in annotation['object']:
        ymin.append(float(obj['bndbox']['ymin'])/height)
        xmin.append(float(obj['bndbox']['xmin'])/width)
        ymax.append(float(obj['bndbox']['ymax'])/height)
        xmax.append(float(obj['bndbox']['xmax'])/width)
        try:
            classes.append(labels[obj['name']])
        except KeyError:
            print(annotation_name, 'has unknown label!')
            sys.exit(1)

    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(encoded_jpg),
        'xmin': _float_list_feature(xmin),
        'xmax': _float_list_feature(xmax),
        'ymin': _float_list_feature(ymin),
        'ymax': _float_list_feature(ymax),
        'labels': _int64_list_feature(classes),
    }))
    return example


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():
    ARGS = make_args()

    with open(ARGS.labels, 'r') as f:
        labels = {line.strip(): i for i, line in enumerate(f.readlines())}
    assert len(labels) > 0
    print('Possible labels:', labels)

    image_dir = ARGS.image_dir
    annotations_dir = ARGS.annotations_dir
    print('Reading images from:', image_dir)
    print('Reading annotations from:', annotations_dir, '\n')

    writer = tf.python_io.TFRecordWriter(ARGS.output)
    examples_list = os.listdir(annotations_dir)
    num_examples = 0
    for example in tqdm(examples_list):
        path = os.path.join(annotations_dir, example)
        annotation = json.load(open(path))
        tf_example = dict_to_tf_example(annotation, image_dir, labels)
        writer.write(tf_example.SerializeToString())
        num_examples += 1

    writer.close()


main()
