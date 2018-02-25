import tensorflow as tf

from src.constants import SHUFFLE_BUFFER_SIZE, PREFETCH_BUFFER_SIZE, NUM_THREADS, RESIZE_METHOD
from .random_image_crop import random_image_crop
from .other_augmentations import random_color_manipulations, random_flip_left_right,\
    random_pixel_value_scale, random_jitter_boxes, random_black_patches


class Pipeline:
    """Input pipeline for training or evaluating object detectors."""

    def __init__(self, filename, batch_size, image_size,
                 repeat=False, shuffle=False, augmentation=None):
        """
        Arguments:
            filename: a string, a path to a tfrecords file.
            batch_size: an integer.
            image_size: a list with two integers [width, height],
                images of this size will be in a batch.
            shuffle: whether to shuffle the dataset.
            augmentation: a dict with parameters or None.
        """
        self.image_width, self.image_height = image_size
        self.augmentation = augmentation
        self.batch_size = batch_size
        self.num_examples = sum(1 for _ in tf.python_io.tf_record_iterator(filename))
        assert self.num_examples > 0

        dataset = tf.data.TFRecordDataset(filename)
        dataset = dataset.repeat(1 if not repeat else None)
        dataset = dataset.map(self._parse_and_preprocess, num_parallel_calls=NUM_THREADS)

        if shuffle:
            dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)

        padded_shapes = ([3, self.image_height, self.image_width], [None, 4], [None], [], [])
        # this works for tf v1.5
        # dataset = dataset.apply(
        #    tf.contrib.data.padded_batch_and_drop_remainder(batch_size, padded_shapes=padded_shapes)
        # )
        is_full_batch = lambda x1, x2, x3, x4, x5: tf.equal(tf.shape(x1)[0], batch_size)
        dataset = dataset.padded_batch(batch_size, padded_shapes).filter(is_full_batch)
        dataset = dataset.prefetch(PREFETCH_BUFFER_SIZE)

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
        images.set_shape([self.batch_size, 3, self.image_height, self.image_width])
        boxes.set_shape([self.batch_size, None, 4])
        labels.set_shape([self.batch_size, None])
        num_boxes.set_shape([self.batch_size])
        filenames.set_shape([self.batch_size])

        features = {'images': images, 'filenames': filenames}
        labels = {'boxes': boxes, 'labels': labels, 'num_boxes': num_boxes}
        return features, labels

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

        # get ground truth boxes, they are in from-zero-to-one format
        boxes = tf.stack([
            parsed_features['ymin'], parsed_features['xmin'],
            parsed_features['ymax'], parsed_features['xmax']
        ], axis=1)
        boxes = tf.to_float(boxes)
        boxes = tf.clip_by_value(boxes, clip_value_min=0.0, clip_value_max=1.0)

        if self.augmentation is not None:
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
        params = self.augmentation

        if params['do_random_crop']:
            image, boxes, labels = random_image_crop(
                image, boxes, labels, probability=0.8,
                min_object_covered=0.0,
                aspect_ratio_range=(0.85, 1.15),
                area_range=(0.333, 0.8),
                overlap_thresh=0.3
            )
        image = tf.image.resize_images(
            image, [self.image_height, self.image_width],
            method=RESIZE_METHOD
        )

        if params['do_random_color_manipulations']:
            image = random_color_manipulations(image, probability=0.7, grayscale_probability=0.07)

        if params['do_random_pixel_scale']:
            image = random_pixel_value_scale(image, minval=0.85, maxval=1.15, probability=0.7)

        if params['do_random_jitter_boxes']:
            boxes = random_jitter_boxes(boxes, ratio=0.05)

        if params['do_random_black_patches']:
            image = random_black_patches(image, max_black_patches=20, probability=0.5, size_to_image_ratio=0.1)

        image, boxes = random_flip_left_right(image, boxes)
        return image, boxes, labels
