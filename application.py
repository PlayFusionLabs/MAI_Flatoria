import os
from PIL import Image
from numpy import asarray
import sys
from decouple import config as env
from numpy import expand_dims
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf

from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mock_data import mock_data
from mrcnn.model import mold_image


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


global _model
global _graph
global cfg
ROOT_DIR = os.path.abspath("./")
WEIGHTS_FOLDER = "./weights"
MODEL_NAME = "mask_rcnn_hq"
WEIGHTS_FILE_NAME = 'maskrcnn_15_epochs.h5'

sys.path.append(ROOT_DIR)



application = Flask(__name__)
cors = CORS(application, resources={r"/*": {"origins": "*"}})


class PredictionConfig(Config):
	NAME = "floorPlan_cfg"
	NUM_CLASSES = 1 + 3
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1


@application.before_first_request
def load_model():
	global cfg
	global _model
	model_folder_path = os.path.abspath("./") + "/mrcnn"
	weights_path = os.path.join(WEIGHTS_FOLDER, WEIGHTS_FILE_NAME)
	cfg = PredictionConfig()
	_model = MaskRCNN(mode='inference', model_dir=model_folder_path, config=cfg)
	_model.load_weights(weights_path, by_name=True)
	global _graph
	_graph = tf.get_default_graph()


def myImageLoader(imageInput):
	image = asarray(imageInput)
	h, w, c = image.shape
	if image.ndim != 3:
		image = skimage.color.gray2rgb(image)
		if image.shape[-1] == 4:
			image = image[..., :3]
	return image, w, h


def getClassNames(classIds):
	result = list()
	for classid in classIds:
		data = {}
		if classid == 1:
			data['name'] = 'wall'
		if classid == 2:
			data['name'] = 'window'
		if classid == 3:
			data['name'] = 'door'
		result.append(data)

	return result


def normalizePoints(bbx, classNames):
	normalizingX = 1
	normalizingY = 1
	result = list()
	doorCount = 0
	index = -1
	doorDifference = 0
	for bb in bbx:
		index = index+1
		if (classNames[index] == 3):
			doorCount = doorCount+1
			if (abs(bb[3]-bb[1]) > abs(bb[2]-bb[0])):
				doorDifference = doorDifference+abs(bb[3]-bb[1])
			else:
				doorDifference = doorDifference+abs(bb[2]-bb[0])

		result.append([bb[0]*normalizingY, bb[1]*normalizingX,
		              bb[2]*normalizingY, bb[3]*normalizingX])
	if doorCount == 0:
		doorCount = 0.01 #TODO
	return result, (doorDifference/doorCount)


def turnSubArraysToJson(objectsArr):
	result = list()
	for obj in objectsArr:
		data = {}
		data['x1'] = obj[1]
		data['y1'] = obj[0]
		data['x2'] = obj[3]
		data['y2'] = obj[2]
		result.append(data)
	return result


@application.route('/', methods=['POST'])
def prediction():
	global cfg
	data = {}
	global _model
	global _graph

	try:
		imagefile = Image.open(request.files['image'].stream)
	except:
		return jsonify(mock_data)

	image, w, h = myImageLoader(imagefile)
	scaled_image = mold_image(image, cfg)
	sample = expand_dims(scaled_image, 0)

	with _graph.as_default():
		r = _model.detect(sample, verbose=0)[0]

	# output_data = model_api(imagefile)

	bbx = r['rois'].tolist()
	temp, averageDoor = normalizePoints(bbx, r['class_ids'])
	temp = turnSubArraysToJson(temp)
	data['points'] = temp
	data['classes'] = getClassNames(r['class_ids'])
	data['Width'] = w
	data['Height'] = h
	data['averageDoor'] = averageDoor

	return jsonify(data)


if __name__ == '__main__':
	application.debug = env('FLASK_DEBUG')
	application.run(host=env('FLASK_HOST'), port=env('FLASK_PORT'), debug=env('FLASK_DEBUG'))
