import tensorflow as tf
from src.constants import SHUFFLE_BUFFER_SIZE, PREFETCH_BUFFER_SIZE,\
    NUM_THREADS, RESIZE_METHOD

from .random_image_crop import random_image_crop
from .other_augmentations import random_color_manipulations, random_flip_left_right,\
    random_pixel_value_scale, random_jitter_boxes, random_black_patches


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filename, batch_size, image_size,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Arguments:
            filename: a string, a path to a tfrecords file.
            batch_size: an integer.
            image_size: a tuple of two integers (width, height),
                images of this size will be in a batch.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        self.image_width, self.image_height = image_size
        self.augmentation = augmentation
        self.num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        assert self.num_examples > 0

        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.map(
            self._parse_and_preprocess,
            num_parallel_calls=NUM_THREADS
        ).prefetch(PREFETCH_BUFFER_SIZE)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        padded_shapes = ([3, self.image_height, self.image_width], [None, 4], [None], [])
        dataset = dataset.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes=padded_shapes)
        )  # make fixed size batches

        if repeat:
            dataset = dataset.repeat()

        self.iterator = tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes
        )
        self.init = self.iterator.make_initializer(dataset)

    def get_batch(self):
        """
        Returns:
            image: a float tensor with shape [batch_size, 3, image_height, image_width].
            boxes: a float tensor with shape [batch_size, max_num_boxes, 4].
            labels: an int tensor with shape [batch_size, max_num_boxes].
            num_boxes: an int tensor with shape [batch_size].
                where max_num_boxes = max(num_boxes).
        """
        images, boxes, labels, num_boxes = self.iterator.get_next()
        batch = {
            'images': images, 'boxes': boxes,
            'labels': labels, 'num_boxes': num_boxes
        }
        return batch

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [3, image_height, image_width], an RGB image
                with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4].
            labels: an int tensor with shape [num_boxes].
            num_boxes: an int tensor with shape [].
        """
        features = {
            'image': tf.FixedLenFeature([], tf.string),
            'ymin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmin': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'ymax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'xmax': tf.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'labels': tf.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        parsed_features = tf.parse_single_example(example_proto, features)

        # get image
        image = tf.image.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # now pixel values are scaled to [0, 1] range

        # get labels
        labels = tf.to_int32(parsed_features['labels'])

        # get ground truth boxes, they are in from-zero-to-one format
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        if self.augmentation:
            image, boxes, labels = _augmentation(image, boxes, labels)

        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        )
        image = tf.transpose(image, perm=[2, 0, 1])  # to NCHW format
        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        return image, boxes, labels, num_boxes


def _augmentation(image, boxes, labels):
    image, boxes, labels = random_image_crop(image, boxes, labels, probability=0.5)
    image = random_color_manipulations(image, probability=0.5, grayscale_probability=0.1)
    image, boxes = random_flip_left_right(image, boxes)
    image = random_pixel_value_scale(image, minval=0.9, maxval=1.1, probability=0.5)
    boxes = random_jitter_boxes(boxes, ratio=0.05)
    image = random_black_patches(image, max_black_patches=10, probability=0.5, size_to_image_ratio=0.1)
    return image, boxes, labels
