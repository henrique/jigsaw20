""" binary focal loss with label_smoothing """
import tensorflow as tf
from tensorflow.keras import backend as K


def focal_loss(gamma=2., pos_weight=1, label_smoothing=0.05):
    """ binary focal loss with label_smoothing """
    def binary_focal_loss(labels, p):
        """ bfl clojure """
        labels = tf.dtypes.cast(labels, dtype=p.dtype)
        if label_smoothing is not None:
            labels = (1 - label_smoothing) * labels + label_smoothing * 0.5

        # Predicted probabilities for the negative class
        q = 1 - p

        # For numerical stability (so we don't inadvertently take the log of 0)
        p = tf.math.maximum(p, K.epsilon())
        q = tf.math.maximum(q, K.epsilon())

        # Loss for the positive examples
        pos_loss = -(q ** gamma) * tf.math.log(p) * pos_weight

        # Loss for the negative examples
        neg_loss = -(p ** gamma) * tf.math.log(q)

        # Combine loss terms
        loss = labels * pos_loss + (1 - labels) * neg_loss

        return loss

    return binary_focal_loss
