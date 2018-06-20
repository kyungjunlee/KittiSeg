"""
Utilize vgg_fcn8 as encoder.
------------------------

The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann

Details: https://github.com/MarvinTeichmann/KittiSeg/blob/master/LICENSE
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_fcn import fcn32_vgg

import tensorflow as tf

import os


def inference(hypes, images, train=True):
    """Build the MNIST model up to where it may be used for inference.

    Args:
      images: Images placeholder, from inputs().
      train: whether the network is used for train of inference

    Returns:
      softmax_linear: Output tensor with the computed logits.
    """
    vgg16_npy_path = os.path.join(hypes['dirs']['data_dir'], 'weights',
                                  "vgg16.npy")
    vgg_fcn = fcn32_vgg.FCN32VGG(vgg16_npy_path=vgg16_npy_path)

    vgg_fcn.wd = hypes['wd']

    num_classes = hypes['arch']['num_classes']
    vgg_fcn.build(images, train=train, num_classes=num_classes, random_init_fc8=True)

    logits = {}

    logits['images'] = images

    if hypes['arch']['fcn_in'] == 'pool5':
        logits['fcn_in'] = vgg_fcn.pool5
    elif hypes['arch']['fcn_in'] == 'fc7':
        logits['fcn_in'] = vgg_fcn.fc7
    else:
        raise NotImplementedError

    logits['feed2'] = vgg_fcn.pool4
    logits['feed4'] = vgg_fcn.pool3

    logits['fcn_logits'] = vgg_fcn.upscore
    # debugging
    # score_fr
    print("Layer name: ", vgg_fcn.score_fr.name)
    print("Layer shape: ", vgg_fcn.score_fr.shape)
    # upscore
    print("Layer name: ", vgg_fcn.upscore.name)
    print("Layer shape: ", vgg_fcn.upscore.shape)


    logits['deep_feat'] = vgg_fcn.pool5
    logits['early_feat'] = vgg_fcn.conv4_3

    # List what variables to save and restore for finetuning
    """
    vars_to_save = {"conv1_1": vgg_fcn.conv1_1, "conv1_2": vgg_fcn.conv1_2,
                    "pool1": vgg_fcn.pool1,
                    "conv2_1": vgg_fcn.conv2_1, "conv2_2": vgg_fcn.conv2_2,
                    "pool2": vgg_fcn.pool2,
                    "conv3_1": vgg_fcn.conv3_1, "conv3_2": vgg_fcn.conv3_2,
                    "conv3_3": vgg_fcn.conv3_3, "pool3": vgg_fcn.pool3,
                    "conv4_1": vgg_fcn.conv4_1, "conv4_2": vgg_fcn.conv4_2,
                    "conv4_3":  vgg_fcn.conv4_3, "pool4": vgg_fcn.pool4,
                    "conv5_1": vgg_fcn.conv5_1, "conv5_2": vgg_fcn.conv5_2,
                    "conv5_3": vgg_fcn.conv5_3, "pool5": vgg_fcn.pool5,
                    "fc6": vgg_fcn.fc6, "fc7": vgg_fcn.fc7}
    """
    """
    vars_to_save = (vgg_fcn.conv1_1, vgg_fcn.conv1_2, vgg_fcn.pool1,
                    vgg_fcn.conv2_1, vgg_fcn.conv2_2, vgg_fcn.pool2,
                    vgg_fcn.conv3_1, vgg_fcn.conv3_2, vgg_fcn.conv3_3, vgg_fcn.pool3,
                    vgg_fcn.conv4_1, vgg_fcn.conv4_2, vgg_fcn.conv4_3, vgg_fcn.pool4,
                    vgg_fcn.conv5_1, vgg_fcn.conv5_2, vgg_fcn.conv5_3, vgg_fcn.pool5,
                    vgg_fcn.fc6, vgg_fcn.fc7)
    """
    vars_to_save = (vgg_fcn.conv1_1, vgg_fcn.conv1_2,
                    vgg_fcn.conv2_1, vgg_fcn.conv2_2,
                    vgg_fcn.conv3_1, vgg_fcn.conv3_2, vgg_fcn.conv3_3,
                    vgg_fcn.conv4_1, vgg_fcn.conv4_2, vgg_fcn.conv4_3, 
                    vgg_fcn.conv5_1, vgg_fcn.conv5_2, vgg_fcn.conv5_3, 
                    vgg_fcn.fc6, vgg_fcn.fc7)

    logits['saving_vars'] = vars_to_save

    return logits
