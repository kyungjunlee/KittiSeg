"""
An implementation of FCN in tensorflow.
------------------------

The MIT License (MIT)

Copyright (c) 2016 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import scipy as scp
import random
from seg_utils import seg_utils as seg


import tensorflow as tf


def _add_identity(hypes, logits):
    with tf.name_scope('decoder'):
        identity = tf.identity(logits)

    return identity


def _add_sigmoid(hypes, logits):
    num_classes = hypes['arch']['num_classes']
    with tf.name_scope('decoder'):
        logits = tf.reshape(logits, (-1, num_classes))
        # epsilon = tf.constant(value=hypes['solver']['epsilon'])
        # logits = logits + epsilon

        sigmoid = tf.nn.sigmoid(logits)

    return sigmoid


def decoder(hypes, logits, train):
    """Apply decoder to the logits.

    Args:
      logits: Logits tensor, float - [batch_size, (Image's shape)].
                [-inf, inf]

    Return:
      logits: the logits are already decoded.
    """
    decoded_logits = {}
    decoded_logits['logits'] = logits['fcn_logits']
    decoded_logits['sigmoid'] = _add_sigmoid(hypes, logits['fcn_logits'])
    # decoded_logits['identity'] = _add_identity(hypes, logits['fcn_logits'])
    """
    Return logits which shape is the same as an input
    TODO: should we input this given 'logits' into a certain layer to do it?
    """
    return decoded_logits


def loss(hypes, decoded_logits, probs):
    """Calculate the loss from the logits and the labels.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      probs: Probability tensor, float - [batch_size].

    Returns:
      loss: Loss tensor of type float.
    """
    logits = decoded_logits['logits']
    with tf.name_scope('loss'):
        # logits = tf.reshape(logits, (-1, 2))
        logits = tf.reshape(logits, (-1, 1))
        # shape = [logits.get_shape()[0], 2]
        epsilon = tf.constant(value=hypes['solver']['epsilon'])
        probs = tf.to_float(tf.reshape(probs, (-1, 1)))

        sigmoid = tf.nn.sigmoid(logits)

        # l2_loss = tf.nn.l2_loss(logits)

        if hypes['loss'] == 'euclidean':
            euclidean_sum = _compute_euclidean_pixel(probs, sigmoid)

        reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES

        weight_loss = tf.add_n(tf.get_collection(reg_loss_col),
                               name='reg_loss')

        total_loss = euclidean_sum + weight_loss

        losses = {}
        losses['total_loss'] = total_loss
        losses['euclidean'] = euclidean_sum
        losses['weight_loss'] = weight_loss

    return losses



def _compute_euclidean_pixel(probs, sigmoid):
    euclidean_sum = tf.reduce_sum(tf.squared_difference(probs, sigmoid))

    return euclidean_sum



def evaluation(hyp, images, probs, decoded_logits, losses, global_step):
    """Evaluate the quality of the logits at predicting the label.

    Args:
      logits: Logits tensor, float - [batch_size, NUM_CLASSES].
      probs: Probs tensor, float - [batch_size], with values in the
        range [0, 1].

    Returns:
      A scalar int32 tensor with the number of examples (out of batch_size)
      that were predicted correctly.
    """
    # For a classifier model, we can use the in_top_k Op.
    # It returns a bool tensor with shape [batch_size] that is true for
    # the examples where the label's is was in the top k (here k=1)
    # of all logits for that example.
    """
    TODO: understand the code below
    """
    eval_list = []
    logits = tf.reshape(decoded_logits['logits'], (-1, 1))
    probs = tf.reshape(probs, (-1, 1))

    """
    pred = tf.argmax(logits, dimension=1)

    negativ = tf.to_float(tf.equal(pred, 0))
    tn = tf.reduce_sum(negativ*probs[:, 0])
    fn = tf.reduce_sum(negativ*probs[:, 1])

    positive = tf.to_float(tf.equal(pred, 1))
    tp = tf.reduce_sum(positive*probs[:, 1])
    fp = tf.reduce_sum(positive*probs[:, 0])
    """
    # eval_list.append(('Acc. ', (tn+tp)/(tn + fn + tp + fp)))
    eval_list.append(('euclidean', losses['euclidean']))
    eval_list.append(('weight_loss', losses['weight_loss']))

    # eval_list.append(('Precision', tp/(tp + fp)))
    # eval_list.append(('True BG', tn/(tn + fp)))
    # eval_list.append(('True Street [Recall]', tp/(tp + fn)))

    return eval_list
