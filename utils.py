import tensorflow as tf


def add_histograms():
    """Add histograms of all trainable variables."""

    summaries = []
    trainable_vars = tf.trainable_variables()

    for v in trainable_vars:
        tf.summary.histogram(v.name[:-2] + '_hist', v)


def add_weight_decay(weight_decay):
    """Add L2 regularization to all (or some) trainable kernel weights."""

    weight_decay = tf.constant(
        weight_decay, tf.float32,
        [], 'weight_decay'
    )

    trainable_vars = tf.trainable_variables()
    kernels = [v for v in trainable_vars if 'weights' in v.name]
    
    for K in kernels:
        x = tf.multiply(weight_decay, tf.nn.l2_loss(K))
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, x)
        
        
class IteratorInitializerHook(tf.train.SessionRunHook):
    """Hook to initialise data iterator after Session is created."""

    def __init__(self):
        super(IteratorInitializerHook, self).__init__()
        self.iterator_initializer_func = None

    def after_create_session(self, session, coord):
        """Initialise the iterator after the session has been created."""
        self.iterator_initializer_func(session)

