import tensorflow as tf
import itertools


"""
Note that for FPN it is required that image height and image width
are divisible by maximal feature stride (by default it is 128).
It is because of all upsampling layers.
"""


class AnchorGenerator:
    def __init__(self, strides=[8, 16, 32, 64, 128],
                 scales=[32, 64, 128, 256, 512],
                 scale_multipliers=[1.0, 1.4142],
                 aspect_ratios=[1.0, 2.0, 0.5]):
        """
        The number of scales and strides must
        be equal to the number of feature maps.

        Note that 1.4142 is equal to sqrt(2).

        So, the number of anchors on each feature map is:
        w * h * len(aspect_ratios) * len(scale_multipliers),
        where (w, h) is the spatial size of the feature map.

        Arguments:
            strides: a list of integers, the feature strides.
            scales: a list of integers, a main scale for each feature map.
            scale_multipliers: a list of floats, a factors for a main scale.
            aspect_ratios: a list of float numbers, aspect ratios to place on each grid point.
        """
        assert len(strides) == len(scales)
        self.strides = strides
        self.scales = scales
        self.scale_multipliers = scale_multipliers
        self.aspect_ratios = aspect_ratios
        self.num_anchors_per_location = len(aspect_ratios) * len(scale_multipliers)

    def __call__(self, image_height, image_width):
        """
        Note that we don't need to pass feature map shapes
        because we use only 'SAME' padding in all our networks.

        Arguments:
            image_height, image_width: scalar int tensors.
        Returns:
            a float tensor with shape [num_anchors, 4],
            boxes with normalized coordinates (and clipped to the unit square).
        """
        with tf.name_scope('anchor_generator'):

            image_height = tf.to_float(image_height)
            image_width = tf.to_float(image_width)

            feature_map_info = []
            num_anchors_per_feature_map = []
            for stride in self.strides:
                h = tf.to_int32(tf.ceil(image_height/stride))
                w = tf.to_int32(tf.ceil(image_width/stride))
                feature_map_info.append((stride, h, w))
                num_anchors_per_feature_map.append(h * w * self.num_anchors_per_location)

            # these are needed elsewhere
            self.num_anchors_per_feature_map = num_anchors_per_feature_map

            anchors = []

            # this is shared by all feature maps
            pairs = list(itertools.product(self.scale_multipliers, self.aspect_ratios))
            aspect_ratios = tf.constant([a for _, a in pairs], dtype=tf.float32)

            for i, (stride, h, w) in enumerate(feature_map_info):

                scales = tf.constant([m * self.scales[i] for m, _ in pairs], dtype=tf.float32)
                stride = tf.constant(stride, dtype=tf.float32)

                """
                It is true that
                image_height = h * stride - x, where 0 <= x < stride.

                Then image_height = (h - 1) * stride + (stride - x).
                So offset y must be equal to 0.5 * (stride - x).

                x = h * stride - image_height,
                y = 0.5 * (image_height - (h - 1) * stride),
                0 < y <= 0.5 * stride.

                Offset y is maximal when image_height is divisible by stride.
                Offset y is minimal when image_height = k * stride + 1, where k is a positive integer.
                """
                offset_y = 0.5 * (image_height - (tf.to_float(h) - 1.0) * stride)
                offset_x = 0.5 * (image_width - (tf.to_float(w) - 1.0) * stride)

                anchors.append(tile_anchors(
                    grid_height=h, grid_width=w,
                    scales=scales, aspect_ratios=aspect_ratios,
                    anchor_stride=(stride, stride),
                    anchor_offset=(offset_y, offset_x)
                ))

        with tf.name_scope('concatenate_normalize_clip'):

            # this is for visualization and debugging only
            self.raw_anchors = anchors

            anchors = tf.concat(anchors, axis=0)

            # convert to the [0, 1] range
            scaler = tf.to_float(tf.stack([
                image_height, image_width,
                image_height, image_width
            ]))
            anchors /= scaler

            # clip to the unit square
            anchors = tf.clip_by_value(anchors, 0.0, 1.0)

        return anchors


def tile_anchors(
        grid_height, grid_width,
        scales, aspect_ratios,
        anchor_stride, anchor_offset):
    """
    It returns boxes in absolute coordinates.

    Arguments:
        grid_height: a scalar int tensor, size of the grid in the y direction.
        grid_width: a scalar int tensor, size of the grid in the x direction.
        scales: a float tensor with shape [N],
            it represents the scale of each box in the basis set.
        aspect_ratios: a float tensor with shape [N],
            it represents the aspect ratio of each box in the basis set.
        anchor_stride: a tuple of float scalar tensors,
            difference in centers between anchors for adjacent grid positions.
        anchor_offset: a tuple of float scalar tensors,
            center of the anchor on upper left element of the grid ((0, 0)-th anchor).
    Returns:
        a float tensor with shape [grid_height * grid_width * N, 4].
    """
    N = tf.size(scales)
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts
    widths = scales * ratio_sqrts
    # widths/heights = aspect_ratios,
    # and scales = sqrt(heights * widths)

    # get a grid of box centers
    y_centers = tf.to_float(tf.range(grid_height)) * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.to_float(tf.range(grid_width)) * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
    # they have shape [grid_height, grid_width]

    centers = tf.stack([y_centers, x_centers], axis=2)
    centers = tf.expand_dims(centers, 2)
    centers = tf.tile(centers, [1, 1, N, 1])
    # shape [grid_height, grid_width, N, 2]

    sizes = tf.stack([heights, widths], axis=1)
    sizes = tf.expand_dims(tf.expand_dims(sizes, 0), 0)
    sizes = tf.tile(sizes, [grid_height, grid_width, 1, 1])
    # shape [grid_height, grid_width, N, 2]

    boxes = tf.concat([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=3)
    # it has shape [grid_height, grid_width, N, 4]
    boxes = tf.reshape(boxes, [-1, 4])
    return boxes
