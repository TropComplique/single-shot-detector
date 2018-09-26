import tensorflow as tf

from detector.constants import SHUFFLE_BUFFER_SIZE, NUM_PARALLEL_CALLS, RESIZE_METHOD
from .random_image_crop import random_image_crop
from .other_augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes,\
    random_black_patches


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, batch_size, is_training, image_size=None):
        """
        Note: when evaluating set batch_size to 1.
        And we don't resize images when evaluating.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            is_training: a boolean.
            image_size: a list with two integers [height, width] or None,
                images of this size will be in a batch.
        """
        if not is_training:
            assert batch_size == 1
            self.image_size = [None, None]
        else:
            assert image_size is not None
            self.image_size = image_size

        def get_num_samples(filename):
            return sum(1 for _ in tf.python_io.tf_record_iterator(filename))

        num_examples = 0
        for filename in filenames:
            num_examples_in_file = get_num_samples(filename)
            assert num_examples_in_file > 0
            num_examples += num_examples_in_file
        self.num_examples = num_examples
        assert self.num_examples > 0

        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        num_shards = len(filenames)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=num_shards)

        dataset = dataset.flat_map(tf.data.TFRecordDataset)
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if is_training else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_PARALLEL_CALLS)

        # we need batches of fixed size
        padded_shapes = (self.image_size + [3], [None, 4], [None], [])
        dataset = dataset.padded_batch(batch_size, padded_shapes, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=1)

        self.dataset = dataset

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [image_height, image_width, 3],
                an RGB image with pixel values in the range [0, 1].
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

        # get groundtruth boxes, they must be in from-zero-to-one format
        boxes = tf.to_float(tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1))

        if self.is_training:
            image, boxes, labels = self.augmentation(image, boxes, labels)

        # we need `num_boxes` because we pad boxes and labels with zeros for batching

        features = {'images': image}
        labels = {'boxes': boxes, 'labels': labels, 'num_boxes': tf.to_int32(tf.shape(boxes)[0])}
        return features, labels

    def augmentation(self, image, boxes, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha

        image, boxes, labels = random_image_crop(
            image, boxes, labels, probability=0.5,
            min_object_covered=0.0,
            aspect_ratio_range=(0.85, 1.15),
            area_range=(0.333, 0.9),
            overlap_thresh=0.3
        )
        image = tf.image.resize_images(image, self.image_size, method=RESIZE_METHOD)

        image = random_color_manipulations(image, probability=0.25, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.25)
        boxes = random_jitter_boxes(boxes, ratio=0.01)
        image = random_black_patches(image, max_patches=10, probability=0.5, size_to_image_ratio=0.1)
        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes, labels
