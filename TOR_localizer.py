"""
objecr localizer based on the pose an location of hand(s)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys

# https://github.com/tensorflow/tensorflow/issues/2034#issuecomment-220820070
import numpy as np
import scipy as scp
import tensorflow as tf

from fractions import Fraction

# configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

sys.path.insert(1, 'incl')

from seg_utils import seg_utils as seg

try:
  # Check whether setup was done correctly
  import tensorvision.utils as tv_utils
  import tensorvision.core as core
except ImportError:
  # You forgot to initialize submodules
  logging.error("Could not import the submodules.")
  logging.error("Please execute:"
                "'git submodule update --init --recursive'")
  exit(1)


class Localizer:
  
  # constructor
  def __init__(self, model = 'GTEA_obj_fcn32_pool5_xentropy_10k_1e-6',
                logdir = None,
                threshold = 0.5,
                image_width = None,
                image_height = None,
                debug = False):
    # flag for debugging mode
    self.debug = debug
    # flag indicating whether the last layer is softmax or not
    self.is_softmax = True
    # NOTE: as of now, if model has "GTEA_ooi", use sigmoid instead
    if "GTEA_ooi" in model:
      self.is_softmax = False

    # set the log directory
    if logdir:
      if self.debug:
        logging.info("Using weights found in {}".format(logdir))
      self.logdir = logdir
    else:
      if 'TV_DIR_RUNS' in os.environ:
        runs_dir = os.path.join(os.environ['TV_DIR_RUNS'],
                                'KittiSeg')
      else:
        runs_dir = 'RUNS'
      self.logdir = os.path.join(runs_dir, model)

    # threshold to filter inference output
    self.threshold = threshold
    
    # resizing parameters
    self.image_width = image_width
    self.image_height = image_height

    # Loading hyperparameters from logdir
    hypes = tv_utils.load_hypes_from_logdir(self.logdir, base_path='hypes')
    logging.info("Hypes loaded successfully.")

    # Loading tv modules (encoder.py, decoder.py, eval.py) from logdir
    modules = tv_utils.load_modules_from_logdir(self.logdir)
    logging.info("Modules loaded successfully. Starting to build tf graph.")

    # Create tf graph and build module.
    with tf.Graph().as_default():
      # Create placeholder for input
      self.input_pl = tf.placeholder(tf.float32)
      image = tf.expand_dims(self.input_pl, 0)

      # build Tensorflow graph using the model from logdir
      self.output_operation = core.build_inference_graph(hypes, modules,
                                                        image=image)
      logging.info("Graph build successfully.")

      # set to allocate memory on GPU as needed
      config = tf.ConfigProto()
      config.gpu_options.allow_growth = True

      # Create a session for running Ops on the Graph.
      self.sess = tf.Session(config = config)
      self.saver = tf.train.Saver()

      # Load weights from logdir
      core.load_weights(self.logdir, self.sess, self.saver)
      logging.info("Weights loaded successfully.")

  """
  Localize an object of interest in an image
  @input  :  image (numpy array)
  @output :  object image (numpy array)
  """
  def do(self, image):
    # read the input image file into numpy array
    # image = scp.misc.imread(image_file, mode = "RGB")
    # resize the input image if set
    if self.image_width is not None and self.image_height is not None:
      image = scp.misc.imresize(image,
                                size = (self.image_height, self.image_width),
                                interp='cubic')
    # start localizing an object of interest 
    feed = {self.input_pl: image}
    # if is_softmax flag is set, using 'softmax'; otherwise, 'sigmoid'
    pred = self.output_operation['softmax'] if self.is_softmax \
            else self.output_operation['sigmoid']
    
    logging.info("Running localizer")
    localizer_output = self.sess.run([pred], feed_dict=feed)

    # reshape output from flat vector to 2D Image
    shape = image.shape
    #print("max prob: ", max(output[0]))
    output_image = localizer_output[0][:, 1] if self.is_softmax else output[0]
    output_image = output_image.reshape(shape[0], shape[1])

    if self.debug:
      scp.misc.imsave("localizerOutput.png", output_image)
      rb_image = seg.make_overlay(image, output_image)
      scp.misc.imsave("localizerRBOutput.png", rb_image)
      logging.info("saved localizer's intermediate results")

    # Accept all pixel with conf >= 0.5 as positive prediction
    # This creates a `hard` prediction result for class street
    logging.info("Given threshold value: %f" % self.threshold)
    logging.info("Max threshold value: %f" % np.max(output_image))

    # getting x y index for drawing a bouding box
    max_index = np.unravel_index(np.argmax(output_image, axis=None),
                                output_image.shape)
    # draw bouding box only if its prob >= threshold
    obj_image = image
    if output_image[max_index[0]][max_index[1]] >= self.threshold:
      # draw bouding box with the x and y
      # NOTE: use the fixed size of bouding box for now 150px x 150px
      # box_color = (0, 255, 0) # green
      # box_line_width = 2
      box_size = 299
      half_box_size = int(box_size / 2)
      # calculate the top-left and the bottom-right
      x1 = max_index[1] - half_box_size if max_index[1] - half_box_size >= 0 else 0
      y1 = max_index[0] - half_box_size if max_index[0] - half_box_size >= 0 else 0
      x2 = x1 + box_size if x1 + box_size <= shape[1] else shape[1]
      y2 = y1 + box_size if y1 + box_size <= shape[0] else shape[0]
      # image with bouding box
      # cv2.rectangle(final_image, (x1, y1), (x2, y2), box_color, box_line_width)
      # object image cropped with the bounding box
      obj_image = obj_image[y1:y2+1, x1:x2+1]

    return obj_image
