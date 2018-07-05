"""
Techable Object Recognizer
- consists of two deep learning models
- 1) object localizer
- 2) object classifier

written by Kyungjun Lee
"""
from TOR_localizer import Localizer
from TOR_classifier import Classifier
# retrain function
from retrain import retrain_model

import cv2



"""
Recognize class
- recognizes an object of interest in an image
- using localizer and classifier
"""
class Recognizer:
  # constructor
  def __init__(self,
              classifier_model,
              classifier_label,
              debug = False):
    # flag for debugging mode
    self.debug = debug
    # initialize Localizer and Classifier
    self.localizer = Localizer(threshold = 0.09,
                              image_width = 720,
                              image_height = 405,
                              debug = debug)
    self.classifier = Classifier(
                        model = classifier_model,
                        label_file = classifier_label,
                        input_layer = "Placeholder",
                        output_layer = "final_result",
                        debug = debug)

  """
  stop the localizer and classifier
  @input  : N/A
  @output : N/A
  """
  def stop_all(self):
    if self.localizer:
      del self.localizer
      self.localizer = None
    
    if self.classifier:
      del self.classifier
      self.classifier = None
    

  """
  resume the localizer and classifier
  @input  : new classifier model file (string),
            new classifier label file (string)
  @output : N/A
  """
  def resume_all(self, classifier_model, classifier_label):
    # re-initialize Localizer and Classifier
    if not self.localizer:
      self.localizer = Localizer(threshold = 0.09,
                                image_width = 720,
                                image_height = 405,
                                debug = self.debug)

    if not self.classifier:
      self.classifier = Classifier(
                          model = classifier_model,
                          label_file = classifier_label,
                          input_layer = "Placeholder",
                          output_layer = "final_result",
                          debug = self.debug)

  """
  recognize an image
  @input  : image_file (RGB file)
  @output : label (string), probability (float)
  """
  def do(self, image):
    print("Start localizing an object of interest")
    output_object = self.localizer.do(image)
    print("Got the localizer output")

    # save the object image if the debugging mode
    if self.debug:
      cv2.imwrite("objectImageFromClient.jpg", output_object)
      print("object image saved to objectImageFromClient.jpg")

    print("Start classifying the object")
    output_label, output_prob = self.classifier.do(output_object)
    print("object: %s, confidence: %.4f" % (output_label, output_prob))

    return output_label, output_prob


  """
  check whether the localizer and classifier alive
  @input  : N/A
  @output : True/False (bool)
  """
  def is_alive(self):
    if self.localizer and self.classifier:
      return True
    else:
      return False


  """
  localize an object of interest in an image
  @input  : image (numpy array)
  @output : object image (numpy array)
  """
  def localize(self, image):
    print("Start localizing an object of interest")
    output_object = self.localizer.do(image)
    print("Got the localizer output")
    return output_object


  """
  retrain the model, especially the classifier
  @input  : N/A
  @output : new classifier model name (string)
  """
  def retrain(self):
    print("Start retraining the model")
    new_classifier_model, new_classifier_label = retrain_model()
    return new_classifier_model, new_classifier_label
