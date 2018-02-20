import numpy as np
import tensorflow as tf


class Box:
    def __init__(self, image, box, score=None):
        """
        Arguments:
            image: a string.
            box: a numpy float array with shape [4].
            score: a float number or None.
        """
        self.image = image
        self.confidence = score
        self.is_matched = False

        # top left corner
        self.ymin = box[0]
        self.xmin = box[1]

        # bottom right corner
        self.ymax = box[2]
        self.xmax = box[3]


class Evaluator:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        assert num_classes > 0
        self._initialize()

    def evaluate(self, iou_threshold=0.5):
        metrics = {}
        for label in range(self.num_classes):
            metrics[label] = evaluate_detector(
                self.groundtruth_by_label_by_image[label],
                self.detections_by_label[label], iou_threshold
            )
        self.metrics = metrics

    def clear(self):
        self._initialize()

    def get_metric_ops(self, images, groundtruth, predictions):

        def update_op(images, gt_boxes, gt_labels, gt_num_boxes, boxes, labels, scores, num_boxes):
            self.add_groundtruth(images, gt_boxes, gt_labels, gt_num_boxes)
            self.add_detections(images, boxes, labels, scores, num_boxes)

        tensors = [
            images, groundtruth['boxes'], groundtruth['labels'], groundtruth['num_boxes'],
            predictions['boxes'], predictions['labels'], predictions['scores'], predictions['num_boxes']
        ]
        update_op = tf.py_func(update_op, tensors, [], stateful=True)

        def evaluate_func():
            self.evaluate()
            self.clear()
        evaluate_op = tf.py_func(evaluate_func, [], [])

        def get_value_func(label=0, measure='ap'):
            def value_func():
                return np.float32(self.metrics[label][measure])
            return value_func

        with tf.control_dependencies([evaluate_op]):
            eval_metric_ops = {
                measure + '_' + str(label): (tf.py_func(get_value_func(label, measure), [], tf.float32), update_op)
                for label in range(self.num_classes) for measure in ['ap', 'best_precision', 'best_recall']
            }
        return eval_metric_ops

    def _initialize(self):
        self.detections_by_label = {label: [] for label in range(self.num_classes)}
        self.groundtruth_by_label_by_image = {label: {} for label in range(self.num_classes)}

    def add_groundtruth(self, images, boxes, labels, num_boxes):
        batch_size = len(images)
        for i, n, image in zip(range(batch_size), num_boxes, images):
            for box, label in zip(boxes[i][:n], labels[i][:n]):
                groundtruth_by_image = self.groundtruth_by_label_by_image[label]
                if image in groundtruth_by_image:
                    groundtruth_by_image[image] += [Box(image, box)]
                else:
                    groundtruth_by_image[image] = [Box(image, box)]

    def add_detections(self, images, boxes, labels, scores, num_boxes):
        """
        Arguments:
            images: a numpy string array with shape [batch_size].
            boxes: a numpy float array with shape [batch_size, N, 4].
            labels: a numpy int array with shape [batch_size, N].
            scores: a numpy float array with shape [batch_size, N].
            num_boxes: a numpy int array with shape [batch_size].
        """
        batch_size = len(images)
        for i, n, image in zip(range(batch_size), num_boxes, images):
            for box, label, score in zip(boxes[i][:n], labels[i][:n], scores[i][:n]):
                self.detections_by_label[label] += [Box(image, box, score)]


def evaluate_detector(groundtruth_by_img, all_detections, iou_threshold=0.5):
    """
    Arguments:
        groundtruth_by_img: a dict of lists with boxes,
            image -> list of groundtruth boxes on the image.
        all_detections: a list of boxes.
        iou_threshold: a float number.
    Returns:
        a dict with four float numbers.
    """

    # each ground truth box is either TP or FN
    n_groundtruth_boxes = 0

    for boxes in groundtruth_by_img.values():
        n_groundtruth_boxes += len(boxes)

    # sort by confidence in decreasing order
    all_detections.sort(key=lambda box: box.confidence, reverse=True)

    n_correct_detections = 0
    n_detections = 0
    precision = [0.0]*len(all_detections)
    recall = [0.0]*len(all_detections)
    confidences = [box.confidence for box in all_detections]

    for k, detection in enumerate(all_detections):

        # each detection is either TP or FP
        n_detections += 1

        if detection.image in groundtruth_by_img:
            groundtruth_boxes = groundtruth_by_img[detection.image]
        else:
            groundtruth_boxes = []

        best_groundtruth_i, max_iou = match(detection, groundtruth_boxes)
        detection.iou = max_iou

        if best_groundtruth_i >= 0 and max_iou >= iou_threshold:
            box = groundtruth_boxes[best_groundtruth_i]
            if not box.is_matched:
                box.is_matched = True
                box.iou = max_iou
                detection.is_matched = True
                detection.type = 'TP'
                n_correct_detections += 1  # increase number of TP
            else:
                detection.is_matched = False
                detection.type = 'FP'
        else:
            detection.is_matched = False
            detection.type = 'FP'

        precision[k] = float(n_correct_detections)/float(n_detections)
        recall[k] = float(n_correct_detections)/float(n_groundtruth_boxes)

    ap = compute_ap(precision, recall)
    best_threshold, best_precision, best_recall = compute_best_threshold(
        precision, recall, confidences
    )
    return {
        'ap': ap, 'best_precision': best_precision,
        'best_recall': best_recall, 'best_threshold': best_threshold
    }


def compute_best_threshold(precision, recall, confidences):
    """
    Arguments:
        precision, recall, confidences: lists of floats of the same length.

    Returns:
        1. a float number, best confidence threshold.
        2. a float number, precision at the threshold.
        3. a float number, recall at the threshold.
    """
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    confidences = np.asarray(confidences)

    diff = np.abs(precision - recall)
    prod = precision*recall
    best_i = np.argmax(prod*(1.0 - diff))
    best_threshold = confidences[best_i]

    return best_threshold, precision[best_i], recall[best_i]


def compute_iou(box1, box2):
    w = (min(box1.xmax, box2.xmax) - max(box1.xmin, box2.xmin)) + 1
    if w > 0:
        h = (min(box1.ymax, box2.ymax) - max(box1.ymin, box2.ymin)) + 1
        if h > 0:
            intersection = w*h
            w1 = box1.xmax - box1.xmin + 1
            h1 = box1.ymax - box1.ymin + 1
            w2 = box2.xmax - box2.xmin + 1
            h2 = box2.ymax - box2.ymin + 1
            union = (w1*h1 + w2*h2) - intersection
            return float(intersection)/float(union)
    return 0.0


def match(detection, groundtruth_boxes):
    """
    Arguments:
        detection: a box.
        groundtruth_boxes: a list of boxes.

    Returns:
        best_i: an integer, index of the best groundtruth box.
        max_iou: a float number.
    """
    best_i = -1
    max_iou = 0.0
    for i, box in enumerate(groundtruth_boxes):
        iou = compute_iou(detection, box)
        if iou > max_iou:
            best_i = i
            max_iou = iou
    return best_i, max_iou


def compute_ap(precision, recall):
    previous_recall_value = 0.0
    ap = 0.0
    # recall is in increasing order
    for p, r in zip(precision, recall):
        delta = r - previous_recall_value
        ap += p*delta
        previous_recall_value = r
    return ap
