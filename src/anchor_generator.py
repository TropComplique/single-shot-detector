import tensorflow as tf
import math


class AnchorGenerator:
    def __init__(self, scales=None, min_scale=0.2, max_scale=0.9,
                 aspect_ratios=(1.0, 2.0, 3.0, 0.5, 0.333),
                 interpolated_scale_aspect_ratio=1.0,
                 reduce_boxes_in_lowest_layer=True):
        """Creates SSD anchors.

        Grid sizes are assumed to be passed in at generation time from finest resolution
        to coarsest resolution --- this is used to (linearly) interpolate scales of anchor boxes.

        Arguments:
            scales: a list of float numbers or None.
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

    def __call__(self, image_features, images):
        """
        Arguments:
            image_features: a list of float tensors where the ith tensor
                has shape [batch, channels_i, height_i, width_i].
            images:  a float tensor with shape [batch_size, 3, height, width].
        Returns:
            a float tensor with shape [num_anchor, 4],
            boxes with normalized coordinates (clipped to the unit square).
        """
        feature_map_shape_list = []
        num_layers = len(image_features)
        for feature_map in image_features:
            height_i, width_i = feature_map.shape.as_list()[2:]
            feature_map_shape_list.append((height_i, width_i))
        h, w = images.shape.as_list()[2:]
        image_aspect_ratio = w/h

        scales = self.scales
        if scales is None:
            scales = [
                self.min_scale + (self.max_scale - self.min_scale)*i/(num_layers - 1)
                for i in range(num_layers)
            ]
        assert len(scales) == num_layers
        scales = scales + [1.0]

        box_specs_list = _get_box_specs(self, scales)

        anchor_grid_list, num_anchors_per_feature_map = [], []
        for grid_size, box_spec in zip(feature_map_shape_list, box_specs_list):
            scales, aspect_ratios = zip(*box_spec)
            h, w = grid_size
            stride = (1.0/tf.to_float(h), 1.0/tf.to_float(w))
            offset = (0.5 * stride[0], 0.5 * stride[1])
            anchor_grid_list.append(tile_anchors(
                image_aspect_ratio, grid_size[0], grid_size[1],
                scales, aspect_ratios, stride, offset
            ))
            num_anchors_per_feature_map.append(grid_size[0] * grid_size[1] * len(scales))

        self.num_anchors_per_location = [len(layer_box_specs) for layer_box_specs in box_specs_list]
        self.feature_map_shape_list = feature_map_shape_list
        self.num_anchors_per_feature_map = num_anchors_per_feature_map
        self.anchor_grid_list = anchor_grid_list

        anchors = tf.concat(anchor_grid_list, axis=0)
        anchors = tf.clip_by_value(anchors, 0.0, 1.0)
        return anchors

    def _get_box_specs(self, scales):
        box_specs_list = []
        for layer, (scale, scale_next) in enumerate(zip(scales[:-1], scales[1:])):
            layer_box_specs = []
            if layer == 0 and self.reduce_boxes_in_lowest_layer:
                layer_box_specs = [(0.1, 1.0), (scale, 2.0), (scale, 0.5)]
            else:
                for aspect_ratio in self.aspect_ratios:
                    layer_box_specs.append((scale, aspect_ratio))
                if self.interpolated_scale_aspect_ratio > 0.0:
                    layer_box_specs.append((math.sqrt(scale*scale_next), self.interpolated_scale_aspect_ratio))
            box_specs_list.append(layer_box_specs)
        return box_specs_list


def tile_anchors(image_aspect_ratio, grid_height, grid_width, scales, aspect_ratios, anchor_stride, anchor_offset):
    """Create a tiled set of anchors strided along a grid in image space.

    Arguments:
        image_aspect_ratio: a float tensor with shape [].
        grid_height: size of the grid in the y direction (an integer or int scalar tensor).
        grid_width: size of the grid in the x direction (an integer or int scalar tensor).
        scales: a float tensor with shape [N],
            it represents the scale of each box in the basis set.
        aspect_ratios: a float tensor with shape [N],
            it represents the aspect ratio of each box in the basis set.
        anchor_stride: a float tensor with shape [2], difference in centers between
            base anchors for adjacent grid positions.
        anchor_offset: a float tensor with shape [2],
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
