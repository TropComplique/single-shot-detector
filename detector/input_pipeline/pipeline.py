import tensorflow as tf

from detector.constants import SHUFFLE_BUFFER_SIZE,\
    NUM_PARALLEL_CALLS, RESIZE_METHOD, DIVISOR
from .random_image_crop import random_image_crop
from .other_augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes,\
    random_black_patches


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, is_training, params):
        """
        During the evaluation we resize images keeping aspect ratio.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            is_training: a boolean.
            params: a dict.
        """
        if not is_training:
            batch_size = 1
            self.image_size = [None, None]
            self.min_dimension = params['min_dimension']
        else:
            batch_size = params['batch_size']
            height = params['image_height']
            width = params['image_width']
            assert height % DIVISOR == 0
            assert width % DIVISOR == 0
            self.image_size = [height, width]

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

        # get an image, it will be decoded after cropping
        image_as_string = parsed_features['image']

        # get labels
        labels = tf.to_int32(parsed_features['labels'])

        # get groundtruth boxes, they must be in from-zero-to-one format
        boxes = tf.to_float(tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1))

        if self.is_training:
            image, boxes, labels = self.augmentation(image_as_string, boxes, labels)
        else:
            image = tf.image.decode_jpeg(image_as_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32)
            # now pixel values are scaled to [0, 1] range

            image, box_scaler = resize_keeping_aspect_ratio(image, self.min_dimension, DIVISOR)
            boxes *= box_scaler

        # we need `num_boxes` because we pad boxes and labels with zeros for batching

        features = {'images': image}
        labels = {'boxes': boxes, 'labels': labels, 'num_boxes': tf.to_int32(tf.shape(boxes)[0])}
        return features, labels

    def augmentation(self, image_as_string, boxes, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha

        image, boxes, labels = random_image_crop(
            image_as_string, boxes, labels, probability=0.7,
            min_object_covered=0.9,
            aspect_ratio_range=(0.9, 1.1),
            area_range=(0.333, 0.9),
            overlap_thresh=0.3
        )
        image = tf.image.resize_images(image, self.image_size, method=RESIZE_METHOD)

        image = random_color_manipulations(image, probability=0.25, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.2)
        boxes = random_jitter_boxes(boxes, ratio=0.01)
        image = random_black_patches(image, max_patches=10, probability=0.2, size_to_image_ratio=0.1)
        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes, labels


def resize_keeping_aspect_ratio(image, min_dimension, divisor):
    """
    When using FPN, divisor must be equal to 128.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        min_dimension: an integer.
        divisor: an integer.
    Returns:
        image: a float tensor with shape [new_height, new_width, 3],
            where `min_dimension = min(new_height, new_width)`,
            `new_height` and `new_width` are divisible by `divisor`.
        box_scaler: a float tensor with shape [4].
    """
    assert min_dimension % divisor == 0

    min_dimension = tf.constant(min_dimension, dtype=tf.int32)
    divisor = tf.constant(divisor, dtype=tf.int32)

    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    original_min_dim = tf.minimum(height, width)
    scale_factor = tf.to_float(min_dimension / original_min_dim)

    def scale(x):
        x = tf.to_int32(tf.ceil(x * scale_factor / divisor))
        return divisor * x

    new_height, new_width = tf.cond(
        tf.greater_equal(height, width),
        lambda: (scale(height), min_dimension),
        lambda: (min_dimension, scale(width))
    )

    image = tf.image.resize_image_with_pad(
        image, new_height, new_width,
        method=RESIZE_METHOD
    )
    # it pads image at the bottom or at the right

    # we need to rescale bounding box coordinates
    box_scaler = tf.to_float(tf.stack([
        height/new_height, width/new_width,
        height/new_height, width/new_width
    ]))

    return image, box_scaler
