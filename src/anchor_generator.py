import tensorflow as tf
import math


class AnchorGenerator:
    def __init__(self, scales=None, min_scale=0.2, max_scale=0.9,
                 aspect_ratios=(1.0, 2.0, 3.0, 0.5, 0.333),
                 interpolated_scale_aspect_ratio=1.0,
                 reduce_boxes_in_lowest_layer=True):
        """Creates SSD anchors.
        Grid sizes are assumed to be passed in at generation
        time from finest resolution to coarsest resolution.

        Arguments:
            scales: a list of float numbers or None,
                if scales is None then min_scale and max_scale are used.
            min_scale: a float number, scale of anchors corresponding to finest resolution.
            max_scale: a float number, scale of anchors corresponding to coarsest resolution.
            aspect_ratios: a list or tuple of float numbers, aspect ratios to place on each grid point.
            interpolated_scale_aspect_ratio: an additional anchor is added with this
                aspect ratio and a scale interpolated between the scale for a layer
                and the scale for the next layer (1.0 for the last layer).
            reduce_boxes_in_lowest_layer: a boolean to indicate whether the fixed 3
                boxes per location is used in the lowest layer.
        """
        self.scales = scales
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.aspect_ratios = aspect_ratios
        self.interpolated_scale_aspect_ratio = interpolated_scale_aspect_ratio
        self.reduce_boxes_in_lowest_layer = reduce_boxes_in_lowest_layer

    def __call__(self, image_features, image_size):
        """
        Arguments:
            image_features: a list of float tensors where the ith tensor
                has shape [batch, channels_i, height_i, width_i].
            image_size: a tuple of integers (width, height).
        Returns:
            a float tensor with shape [num_anchor, 4],
            boxes with normalized coordinates (and clipped to the unit square).
        """
        feature_map_shape_list = []
        num_layers = len(image_features)
        for feature_map in image_features:
            height_i, width_i = feature_map.shape.as_list()[2:]
            feature_map_shape_list.append((height_i, width_i))
        w, h = image_size
        image_aspect_ratio = w/h

        scales = self.scales
        if scales is None:
            scales = [
                self.min_scale + (self.max_scale - self.min_scale)*i/(num_layers - 1)
                for i in range(num_layers)
            ]
        assert len(scales) == num_layers
        scales = scales + [1.0]
        box_specs_list = self._get_box_specs(scales)
        # number of anchors per cell in a grid
        self.num_anchors_per_location = [len(layer_box_specs) for layer_box_specs in box_specs_list]

        with tf.name_scope('anchor_generator'):
            anchor_grid_list, num_anchors_per_feature_map = [], []
            for grid_size, box_spec in zip(feature_map_shape_list, box_specs_list):
                scales, aspect_ratios = zip(*box_spec)
                h, w = grid_size
                stride = (1.0/tf.to_float(h), 1.0/tf.to_float(w))
                offset = (0.5/tf.to_float(h), 0.5/tf.to_float(w))
                anchor_grid_list.append(tile_anchors(
                    image_aspect_ratio=image_aspect_ratio,
                    grid_height=h, grid_width=w, scales=scales,
                    aspect_ratios=aspect_ratios, anchor_stride=stride,
                    anchor_offset=offset
                ))
                num_anchors_per_feature_map.append(h * w * len(scales))

        # constant tensors, anchors for each feature map
        self.anchor_grid_list = anchor_grid_list
        self.num_anchors_per_feature_map = num_anchors_per_feature_map

        with tf.name_scope('concatenate'):
            anchors = tf.concat(anchor_grid_list, axis=0)
            anchors = tf.clip_by_value(anchors, 0.0, 1.0)
            return anchors

    def _get_box_specs(self, scales):
        """
        Arguments:
            scales: a list of floats, it has length num_layers + 1.
        Returns:
            a list of lists of tuples (scale, aspect ratio).
        """
        box_specs_list = []
        for layer, (scale, scale_next) in enumerate(zip(scales[:-1], scales[1:])):
            layer_box_specs = []
            if layer == 0 and self.reduce_boxes_in_lowest_layer:
                layer_box_specs = [(scale, 1.0), (scale, 2.0), (scale, 0.5)]
            else:
                for aspect_ratio in self.aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                if self.interpolated_scale_aspect_ratio > 0.0:
                    layer_box_specs.append((math.sqrt(scale*scale_next), self.interpolated_scale_aspect_ratio))
            box_specs_list.append(layer_box_specs)
        return box_specs_list


def tile_anchors(
        image_aspect_ratio, grid_height, grid_width,
        scales, aspect_ratios, anchor_stride, anchor_offset):
    """
    Arguments:
        image_aspect_ratio: a float tensor with shape [].
        grid_height: an integer, size of the grid in the y direction.
        grid_width: an integer, size of the grid in the x direction.
        scales: a float tensor with shape [N],
            it represents the scale of each box in the basis set.
        aspect_ratios: a float tensor with shape [N],
            it represents the aspect ratio of each box in the basis set.
        anchor_stride: a tuple of float numbers, difference in centers between
            anchors for adjacent grid positions.
        anchor_offset: a tuple of float numbers,
            center of the anchor on upper left element of the grid ((0, 0)-th anchor).
    Returns:
        a float tensor with shape [N * grid_height * grid_width, 4].
    """
    N = tf.size(scales)
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = (scales / ratio_sqrts) * tf.sqrt(image_aspect_ratio)
    widths = (scales * ratio_sqrts) / tf.sqrt(image_aspect_ratio)
    # assume that size(image) = (original_width, original_height) then it must be that
    # image_aspect_ratio = original_width/original_height, and
    # (widths * original_width)/(heights * original_height) = aspect_ratios, and
    # scales = sqrt(heights * widths)

    # get a grid of box centers
    y_centers = tf.to_float(tf.range(grid_height))
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.to_float(tf.range(grid_width))
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
    # they have shape [grid_height, grid_width]

    centers = tf.stack([y_centers, x_centers], axis=2)
    centers = tf.expand_dims(centers, 2)
    centers = tf.tile(centers, [1, 1, N, 1])
    # shape [grid_height, grid_width, N, 2]

    sizes = tf.stack([heights, widths], axis=1)
    sizes = tf.expand_dims(sizes, 0)
    sizes = tf.expand_dims(sizes, 0)
    sizes = tf.tile(sizes, [grid_height, grid_width, 1, 1])
    # shape [grid_height, grid_width, N, 2]

    boxes = tf.concat([centers - 0.5 * sizes, centers + 0.5 * sizes], axis=3)
    # it has shape [grid_height, grid_width, N, 4]
    boxes = tf.reshape(boxes, [-1, 4])
    return boxes
