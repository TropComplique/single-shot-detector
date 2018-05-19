import tensorflow as tf

from src.constants import SHUFFLE_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD,\
    CYCLE_LENGTH, BLOCK_LENGTH
from src.input_pipeline.random_image_crop import random_image_crop
from src.input_pipeline.other_augmentations import random_color_manipulations,\
    random_flip_left_right, random_pixel_value_scale, random_jitter_boxes,\
    random_black_patches


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filenames, batch_size, image_size,
                 repeat=False, shuffle=False, augmentation=False):
        """
        Note: when evaluating set batch_size to 1.

        Arguments:
            filenames: a list of strings, paths to tfrecords files.
            batch_size: an integer.
            image_size: a list with two integers [width, height],
                images of this size will be in a batch.
            repeat: a boolean, whether repeat indefinitely.
            shuffle: whether to shuffle the dataset.
            augmentation: whether to do data augmentation.
        """
        self.image_width, self.image_height = image_size
        self.augmentation = augmentation
        self.batch_size = batch_size

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

        # you cannot set cycle_length too hight
        cycle_length = min(CYCLE_LENGTH, num_shards)

        dataset = dataset.interleave(
            lambda filename: tf.data.TFRecordDataset(filename),
            cycle_length=cycle_length, block_length=BLOCK_LENGTH
        )
        dataset = dataset.prefetch(buffer_size=batch_size)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        dataset = dataset.repeat(None if repeat else 1)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        # we need batches of fixed size
        padded_shapes = ([3, self.image_height, self.image_width], [None, 4], [None], [], [])
        dataset = dataset.apply(
           tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes)
        )
        dataset = dataset.prefetch(buffer_size=1)

        self.iterator = tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes
        )
        self.init = self.iterator.make_initializer(dataset)

    def get_batch(self):
        """
        Returns:
            features: a dict with the following keys
                'images': a float tensor with shape [batch_size, 3, image_height, image_width].
                'filenames': a string tensor with shape [batch_size].
            labels: a dict with the following keys
                'boxes': a float tensor with shape [batch_size, max_num_boxes, 4].
                'labels': an int tensor with shape [batch_size, max_num_boxes].
                'num_boxes': an int tensor with shape [batch_size].
            where max_num_boxes = max(num_boxes).
        """
        images, boxes, labels, num_boxes, filenames = self.iterator.get_next()
        features = {'images': images, 'filenames': filenames}
        labels = {'boxes': boxes, 'labels': labels, 'num_boxes': num_boxes}
        return features, labels

    def _parse_and_preprocess(self, example_proto):
        """What this function does:
        1. Parses one record from a tfrecords file and decodes it.
        2. (optionally) Augments it.

        Returns:
            image: a float tensor with shape [3, image_height, image_width],
                an RGB image with pixel values in the range [0, 1].
            boxes: a float tensor with shape [num_boxes, 4].
            labels: an int tensor with shape [num_boxes].
            num_boxes: an int tensor with shape [].
            filename: a string tensor with shape [].
        """
        features = {
            'filename': tf.FixedLenFeature([], tf.string),
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
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        if self.augmentation:
            image, boxes, labels = self._augmentation_fn(image, boxes, labels)
        else:
            image = tf.image.resize_images(
                image, [self.image_height, self.image_width],
                method=RESIZE_METHOD
            )

        image = tf.transpose(image, perm=[2, 0, 1])  # to NCHW format
        num_boxes = tf.to_int32(tf.shape(boxes)[0])
        filename = parsed_features['filename']
        return image, boxes, labels, num_boxes, filename

    def _augmentation_fn(self, image, boxes, labels):
        # there are a lot of hyperparameters here,
        # you will need to tune them all, haha

        image, boxes, labels = random_image_crop(
            image, boxes, labels, probability=0.5,
            min_object_covered=0.0,
            aspect_ratio_range=(0.85, 1.15),
            area_range=(0.333, 0.9),
            overlap_thresh=0.3
        )
        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        )
        # if you do color augmentations before resizing, it will be very slow!

        image = random_color_manipulations(image, probability=0.25, grayscale_probability=0.05)
        image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.25)
        boxes = random_jitter_boxes(boxes, ratio=0.01)
        image = random_black_patches(image, max_patches=10, probability=0.5, size_to_image_ratio=0.1)
        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes, labels
