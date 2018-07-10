"""
object classifier, using Inception V3
"""
import tensorflow as tf

import numpy as np

class Classifier:

  # constructor
  def __init__(self,
              model = "models/classifier_graph.pb",
              label_file = "models/classifier_labels.txt",
              input_height = 299,
              input_width = 299,
              input_mean = 0,
              input_std = 255,
              input_layer = "input",
              output_layer = "InceptionV3/Predictions/Reshape_1",
              debug = False):
    # flag for debugging mode
    self.debug = debug
    # set classifier model
    self.model = model
    # set label file
    self.label_file = label_file
    # initialize input variables
    self.input_height = input_height
    self.input_width = input_width
    self.input_mean = input_mean
    self.input_std = input_std
    # set layer for input and output
    self.input_layer = input_layer
    self.output_layer = output_layer

    # session for calculating tensor for an input image
    # self.t_sess = tf.Session()

    # load the object classifier model
    obj_recog_graph = self.load_graph(self.model)
    input_name = "import/" + self.input_layer
    output_name = "import/" + self.output_layer
    self.input_operation = obj_recog_graph.get_operation_by_name(input_name)
    self.output_operation = obj_recog_graph.get_operation_by_name(output_name)

    # set to allocate memory on GPU as needed
    # For more details, look at
    # https://stackoverflow.com/questions/36927607/how-can-i-solve-ran-out-of-gpu-memory-in-tensorflow
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # session for recognizer
    self.sess = tf.Session(graph=obj_recog_graph, config = config)


  # function to load graph
  def load_graph(self, model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
      graph_def.ParseFromString(f.read())
    with graph.as_default():
      tf.import_graph_def(graph_def)

    return graph


  # function to read tensor from an image
  def read_tensor_from_image(self, input_image,
                            input_height=299,
                            input_width=299,
                            input_mean=0,
                            input_std=255):

    float_caster = tf.cast(input_image, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)
    # result = self.t_sess.run(normalized)

    return result


  # function to load labels
  def load_labels(self, label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
      label.append(l.rstrip())
    return label


  """
  Classify an image
  @input  :  image (numpy array?)
  @output :  label (string) and confidence (string)
  """
  def do(self, image):
    # now pass the object image to the object classifier
    t = self.read_tensor_from_image(
          image,
          input_height = self.input_height,
          input_width = self.input_width,
          input_mean = self.input_mean,
          input_std = self.input_std)

    results = self.sess.run(
                self.output_operation.outputs[0], {
                self.input_operation.outputs[0]: t
              })
    results = np.squeeze(results)

    top_k = results.argsort()[-1:][::-1]
    labels = self.load_labels(self.label_file)

    for i in top_k:
      print(labels[i], results[i])

    return labels[top_k[0]], results[top_k[0]]

