"""
Techable Object Recogznier Server
- receive images from a client via RESTful protocol
- recognize them
- send back to the client recognition results

written by Jonggi Hong
modified by Kyungjun Lee
"""
from TOR import Recognizer

# code mostly adapted from
# https://gist.github.com/kylehounslow/767fb72fde2ebdd010a0bf4242371594
from flask import Flask, request, jsonify#, Response, json

import numpy as np
import cv2

import os

# configuration for Flask upload
SAVE_DIR = "testing"
IMAGE_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# initializer the Flask application
receiver = Flask(__name__)

"""
initialize recognizer
@input  : classifier model name (string),
          classifier label name (string),
          debugging flag (bool)
@output : recognizer instance (object)
"""
def init_recognizer(classifer_model, classifier_label, debug):
  return Recognizer(classifier_model = classifier_model,
                    classifier_label = classifier_label,
                    debug = debug)


"""
stop the recognizer
@input  : N/A
@output : N/A
"""
def stop_recognizer():
  global recognizer
  # we may have to rely on python's GCC
  recognizer.stop_all()


"""
resume the recognizer
@input  : N/A
@output : N/A
"""
def resume_recognizer(classifier_model, classifier_label):
  global recognizer
  # we may have to rely on python's GCC
  recognizer.resume_all(classifier_model, classifier_label)


"""
reload the latest recognizer
@input  :
@output :
"""
def reload_recognizer(classifier_model, classifier_label):
  new_c_model, new_c_label = get_latest_classifier(classifier_model_dir)
  if classifier_model < new_c_model and classifier_label < new_c_label:
    print("found the latest classifier: %s and %s" % (classifier_model, classifier_label))
    stop_recognizer()
    classifier_model = new_c_model
    classifier_label = new_c_label
    resume_recognizer(classifier_model, classifier_label)


"""
pick the latest version of classifier
@input  : model dir (string)
@output : classifier model dir (string), classifier label dir (string)
"""
def get_latest_classifier(model_dir):
  # get all .pb files in the model dir
  latest_model = latest_label = ""
  for each in os.listdir(model_dir):
    if each.endswith(".pb"):
      if latest_model == "":
        latest_model = each
      elif latest_model < each:
        latest_model = each
    elif each.endswith(".txt"):
      if latest_label == "":
        latest_label = each
      elif latest_label < each:
        latest_model = each

  # get their paths
  latest_model = os.path.join(model_dir, latest_model)
  latest_label = os.path.join(model_dir, latest_label)

  return latest_model, latest_label


"""
check whether a received file is allowed
@input  : filename (string)
@output : True/False (boolean)
"""
def is_image(filename):
  return '.' in filename and \
          filename.rsplit('.', 1)[1].lower() in IMAGE_EXTENSIONS


"""
count the number of images with label
@input  : label (string)
@output : # of images in the label path (int), \
          label directory (string)
"""
def count_images_with_label(label):
  # a folder for the given label
  label_dir = os.path.join(SAVE_DIR, label)

  # create the folder if not existed
  if not os.path.exists(label_dir):
    os.makedirs(label_dir)
    if debug: print("created ", label_dir)

  # count how many images are there
  num_imgs = len([name for name in os.listdir(label_dir) \
              if is_image(name)])

  if debug: print("total: %d in %s" % (num_imgs, label_dir))

  return num_imgs, label_dir


"""
save an image for retraining
@input  : image (numpy array), label (string)
@output : # of images for the label (int)
"""
def save_image_with_label(image, label):
  # first count how many images are there
  num_imgs, label_dir = count_images_with_label(label)

  # image file name: label_num.jpg
  img_name = label + "_" + str(num_imgs) + ".jpg"
  # save the image
  cv2.imwrite(os.path.join(label_dir, img_name), image)
  if debug: print("%s saved in %s" % (img_name, label_dir))

  # increment num_imgs and return
  return num_imgs + 1


"""
receive an image from a client to save
@input  : N/A
@output : Response {count of images in the label}
"""
# route http posts to this method
@receiver.route('/save', methods = ['POST'])
def save_image():
  global input_label
  label = input_label
  r = request
  print("request:", r)
  # convert string of image data to uint8
  raw = np.fromstring(r.data, np.uint8)
  # decode image
  image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
  # get object image
  obj_img = localize_object(image)

  if not label:
    print("Error: no label")
    return jsonify(label = "N/A", count = "N/A")

  # save the image
  cnt = save_image_with_label(obj_img, label)
  
  # build JSON response containing the output label and probability
  return jsonify(label = label, count = str(cnt))


"""
receive a label from a client for retraining
@input  : N/A
@output : Response {count of images in the label}
"""
# route http posts to this method
@receiver.route('/label', methods = ['POST'])
def save_label():
  global input_label

  r = request
  print("request:", r)
  # get json body
  body = r.json
  if not body:
    print("Error: invalid request")
    return jsonify(label = "N/A", count = "N/A")

  label = body['label']
  if not label:
    print("Error: no label")
    return jsonify(label = "N/A", count = "N/A")

  # get the label
  input_label = label

  # save the image
  cnt, _ = count_images_with_label(label)
  
  # build JSON response containing the output label and probability
  return jsonify(label = label, count = str(cnt))


"""
check recognizer
  1) if not initialized, initialize it
  2) if not resumed, resume it
@input  : N/A
@output : N/A
"""
def check_recognizer():
  # get the global recognizer variable
  global recognizer, classifier_model, classifier_label
  # if turned off, turn it on
  if not recognizer:
    recognizer = init_recognizer(classifier_model, classifier_label, debug)
  # check whether recognizer is available
  elif not recognizer.is_alive():
    """
    # get the latest classifier model and label
    new_c_model, new_c_label = get_latest_classifier(classifier_model_dir)
    if new_c_model > classifier_model and new_c_label > classifier_label:
      classifier_model = new_c_model
      classifier_label = new_c_label
    """
    # now resuming it
    resume_recognizer(classifier_model, classifier_label)




"""
localize an object of interest in image
@input  : image (numpy array)
@output : object image (numpy array)
"""
def localize_object(image):
  check_recognizer()

  # localize an object of interest
  obj_img = recognizer.localize(image)
  return obj_img

"""
recognize an image
@input  : image (numpy array)
@output : label (string), probability (float)
"""
def recognize(image):
  check_recognizer()

  # recognize the input image
  label, prob = recognizer.do(image)
  return label, prob


"""
receive an image from a client to recognize
@input  : N/A
@output : Response {label, probability}
"""
# route http posts to this method
@receiver.route('/recognize', methods = ['POST'])
def recognize_image():
  r = request
  print("request:", r)
  # convert string of image data to uint8
  raw = np.fromstring(r.data, np.uint8)
  # decode image
  image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
  # print(image)
  # save the image
  cv2.imwrite("imageFromClient.jpg", image)
  print("image received and saved")
  """ #testing
  label = "N/A"
  prob = 0.
  """
  # recognize image
  label, prob = recognize(image)
  # build JSON response containing the output label and probability
  return jsonify(label = label, prob = str("%.4f" % (prob)))

  """
  response = {'label': label, 'prob': prob}

  return Response(response = json.dumps(response_pickled),
                  status = 200,
                  mimetype = "application/json")
  """


"""
retrain classifiers with images received from a client
@input  : N/A
@output : Response {label, probability}
"""
# route http posts to this method
@receiver.route('/retrain', methods = ['POST'])
def retrain_classifier():
  global recognizer
  # stop the recognizer if alive
  stop_recognizer()

  r = request
  print("request:", r)
  # get json body
  body = r.json
  if not body:
    print("Error: invalid request")
    return jsonify(classifier_model = "N/A", classifier_label = "N/A")

  retrain = body['retrain']
  if not retrain:
    print("Error: no label")
    return jsonify(classifier_model = "N/A", classifier_label = "N/A")

  if retrain == "Yes":
    # trigger the retrain
    new_model, new_label = recognizer.retrain()
    if new_model != "" and new_label != "":
      global classifier_model, classifier_label
      classifier_model = new_model
      classifier_label = new_label

  # build JSON response containing the output label and probability
  return jsonify(classifier_model = classifier_model,
                classifier_label = classifier_label)


# global variables
input_label = None
debug = True
classifier_model_dir = "models/"
# classifier_model, classifier_label = get_latest_classifier(classifier_model_dir)
# recognizer = None
classifier_model = os.path.join(classifier_model_dir, "classifier_graph.pb")
classifier_label = os.path.join(classifier_model_dir, "classifier_labels.txt")
recognizer = init_recognizer(classifier_model, classifier_label, debug)
# stop_recognizer()

# run the RESTful server
receiver.run(host = "0.0.0.0", port = 5000)

