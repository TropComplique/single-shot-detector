import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf

"""
It's supposed to be a tool for visualizing images
with predicted bounding boxes while evaluating.
But it isn't working yet.
Apparently, it is very hard to implement it
with tf.estimator framework.
"""


class ImageVisualizer:
    def __init__(self, num_images_to_show=50):
        self.images = []
        self.num_images_to_show = num_images_to_show

    def get_op(self, images, predictions):
        """
        Arguments:
            images: a float tensor with shape [1, 3, height, width],
                a RGB image with pixel values in the range [0, 1].
            predictions: a dict of tensors.
        Returns:
            a summary op.
        """

        image = tf.transpose(images[0], perm=[1, 2, 0])  # to 'NHWC' format
        num_boxes = predictions['num_boxes'][0]
        tensors = [
            image,
            predictions['boxes'][0][:num_boxes],
            predictions['scores'][0][:num_boxes],
            predictions['labels'][0][:num_boxes]
        ]

        # for now, labels aren't used
        def add_images(image, boxes, scores, labels):
            if len(self.images) < self.num_images_to_show:
                image = (255.0*image).astype('uint8')
                image = draw_boxes_on_image(image, boxes, scores)
                self.images.append(image)
        update_op = tf.py_func(add_images, tensors, [], stateful=True)

        def get_all_images():
            return np.stack(self.images, axis=0)

        with tf.control_dependencies([update_op]):
            all_images = tf.py_func(get_all_images, [], tf.uint8)
            image_summary = tf.summary.image('predictions', all_images, max_outputs=self.num_images_to_show)

        return image_summary


def draw_boxes_on_image(image, boxes, scores):
    """
    Arguments:
        image: a numpy uint8 array with shape [height, width, 3], a RGB image.
        boxes: a numpy float array with shape [N, 4].
        scores: a numpy float array with shape [N].
    Returns:
        a numpy uint8 array with shape [height, width, 3].
    """
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    scale = np.array([height, width, height, width], dtype='float32')
    boxes = boxes*scale

    for box, score in zip(boxes, scores):
        ymin, xmin, ymax, xmax = box
        text = '{0:.3f}'.format(score)
        fill = (255, 255, 255, 45)
        outline = 'red'
        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)],
            fill=fill, outline=outline
        )
        draw.rectangle(
            [(xmin, ymin), (xmin + 6*len(text) + 1, ymin + 12)],
            fill='white', outline='white'
        )
        draw.text((xmin + 1, ymin + 1), text, fill='red')

    return np.array(image, dtype='uint8')
