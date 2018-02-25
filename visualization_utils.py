import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


def get_image_visualizer(images, predictions, max_outputs=64):
    """
    Arguments:
        images: a float tensor with shape [batch_size, 3, height, width],
            a batch of RGB images with pixels values in the range [0, 1].
        predictions:
        max_outputs:
    Returns:
        a summary op.
    """
    images = tf.transpose(images, perm=[0, 2, 3, 1])  # to 'NHWC' format
    tensors = [
        images[:max_outputs],
        predictions['boxes'][:max_outputs],
        predictions['scores'][:max_outputs],
        predictions['labels'][:max_outputs],
        predictions['num_boxes'][:max_outputs]
    ]

    def draw_boxes(images, boxes, scores, labels, num_boxes):
        result = []
        for i, b, s, n in zip(images, boxes, scores, num_boxes):
            result.append(draw_boxes_on_image(i, b[:n], s[:n]))
        return np.stack(result, axis=0)

    images_with_boxes = tf.py_func(draw_boxes, tensors, tf.uint8, stateful=False)
    return tf.summary.image('predictions', images_with_boxes, max_outputs)


def draw_boxes_on_image(image, boxes, scores):
    image = (255.0*image).astype('uint8')
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
            [(xmin, ymin), (xmin + 4*len(text) + 4, ymin + 10)],
            fill='white', outline='white'
        )
        draw.text((xmin + 1, ymin + 1), text, fill='red')

    return np.array(image)
