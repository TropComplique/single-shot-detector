import tensorflow as tf


def random_color_manipulations(image, probability=0.5, grayscale_probability=0.1):

    def manipulate(image):
        image = tf.image.random_brightness(image, 0.1)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    def to_grayscale(image):
        image = tf.image.rgb_to_grayscale(image)
        image = tf.image.grayscale_to_rgb(image)
        return image

    with tf.name_scope('random_color_manipulations'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: manipulate(image), lambda: image)

    with tf.name_scope('to_grayscale'):
        make_gray = tf.less(tf.random_uniform([]), grayscale_probability)
        image = tf.cond(make_gray, lambda: to_grayscale(image), lambda: image)

    return image


def random_flip_left_right(image, boxes):

    def flip(image, boxes):
        flipped_image = tf.image.flip_left_right(image)
        ymin, xmin, ymax, xmax = tf.unstack(boxes, axis=1)
        flipped_xmin = tf.subtract(1.0, xmax)
        flipped_xmax = tf.subtract(1.0, xmin)
        flipped_boxes = tf.stack([ymin, flipped_xmin, ymax, flipped_xmax], 1)
        return flipped_image, flipped_boxes

    with tf.name_scope('random_flip_left_right'):
        do_it = tf.less(tf.random_uniform([]), 0.5)
        image, boxes = tf.cond(do_it, lambda: flip(image, boxes), lambda: (image, boxes))
        return image, boxes


def random_pixel_value_scale(image, minval=0.9, maxval=1.1, probability=0.5):
    """This function scales each pixel independently of the other ones.

    Arguments:
        image: a float tensor with shape [height, width, 3],
            an image with pixel values varying between [0, 1].
        minval: a float number, lower ratio of scaling pixel values.
        maxval: a float number, upper ratio of scaling pixel values.
        probability: a float number.
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def random_value_scale(image):
        color_coef = tf.random_uniform(
            tf.shape(image), minval=minval,
            maxval=maxval, dtype=tf.float32
        )
        image = tf.multiply(image, color_coef)
        image = tf.clip_by_value(image, 0.0, 1.0)
        return image

    with tf.name_scope('random_pixel_value_scale'):
        do_it = tf.less(tf.random_uniform([]), probability)
        image = tf.cond(do_it, lambda: random_value_scale(image), lambda: image)
        return image


def random_jitter_boxes(boxes, ratio=0.05):
    """Randomly jitter bounding boxes.

    Arguments:
        boxes: a float tensor with [N, 4].
        ratio: a float number.
            The ratio of the box width and height that the corners can jitter.
            For example if the width is 100 pixels and ratio is 0.05,
            the corners can jitter up to 5 pixels in the x direction.
    Returns:
        a float tensor with [N, 4].
    """
    def random_jitter_box(box, ratio):
        """Randomly jitter a box.
        Arguments:
            box: a float tensor with shape [1, 1, 4].
            ratio: a float number.
        Returns:
            a float tensor with shape [1, 1, 4].
        """
        rand_numbers = tf.random_uniform(
            [1, 1, 4], minval=-ratio, maxval=ratio, dtype=tf.float32
        )
        ymin, xmin, ymax, xmax = [box[0, 0, i] for i in range(4)]
        box_height, box_width = ymax - ymin, xmax - xmin
        hw_coefs = tf.stack([box_height, box_width, box_height, box_width])
        hw_rand_coefs = tf.multiply(hw_coefs, rand_numbers)
        jittered_box = tf.add(box, hw_rand_coefs)
        jittered_box = tf.clip_by_value(jittered_box, 0.0, 1.0)
        return jittered_box

    with tf.name_scope('random_jitter_boxes'):
        boxes = tf.expand_dims(boxes, 1)
        boxes = tf.expand_dims(boxes, 2)  # shape [N, 1, 1, 4]
        distorted_boxes = tf.map_fn(
            lambda x: random_jitter_box(x, ratio),
            boxes, dtype=tf.float32
        )
        distorted_boxes = tf.squeeze(distorted_boxes, axis=[1, 2])
        return distorted_boxes


def random_black_patches(image, max_black_patches=10, probability=0.5, size_to_image_ratio=0.1):
    """Randomly adds some black (colored) patches to the image.

    Arguments:
        image: a float tensor with shape [height, width, 3].
        max_black_patches: an integer, number of times that the
            function tries to add a black box to the image.
        probability: at each try, what is the chance of adding a box.
        size_to_image_ratio: determines the ratio of the size of the black patches
            to the size of the image. box_size = size_to_image_ratio * min(width, height).
    Returns:
        a float tensor with shape [height, width, 3].
    """
    def add_black_patch_to_image(image):
        image_shape = tf.shape(image)
        image_height, image_width = image_shape[0], image_shape[1]
        box_size = tf.to_int32(tf.multiply(
            tf.minimum(tf.to_float(image_height), tf.to_float(image_width)),
            size_to_image_ratio
        ))
        normalized_y_min = tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio))
        normalized_x_min = tf.random_uniform([], minval=0.0, maxval=(1.0 - size_to_image_ratio))
        y_min = tf.to_int32(normalized_y_min * tf.to_float(image_height))
        x_min = tf.to_int32(normalized_x_min * tf.to_float(image_width))

        random_color = tf.random_uniform([1, 1, 3], minval=-2.0, maxval=0.5, dtype=tf.float32)
        black_box = tf.tile(random_color, [box_size, box_size, 1])
        # black_box = tf.ones([box_size, box_size, 3], dtype=tf.float32)
        black_box_padded = tf.image.pad_to_bounding_box(black_box, y_min, x_min, image_height, image_width)        
        mask = 1.0 - black_box_padded
        image = tf.multiply(image, mask)
        return image

    with tf.name_scope('random_black_patches'):
        for _ in range(max_black_patches):
            do_it = tf.less(tf.random_uniform([]), probability)
            image = tf.cond(do_it, lambda: add_black_patch_to_image(image), lambda: image)

        image = tf.clip_by_value(image, 0.0, 1.0)
        return image
