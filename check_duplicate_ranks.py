# Pyrthon code to generate vectors with Inception model labels for set of images 
from __future__ import absolute_import, division, print_function

import os.path
import re
import sys
import tarfile
import glob
import json
import psutil
from collections import defaultdict
import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
		'model_dir', '/tmp/imagenet',
		"""Path to classify_image_graph_def.pb, """
		"""imagenet_synset_to_human_label_map.txt, and """
		"""imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '', 
		"""Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 7,
		"""Display this many predictions.""")


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'


class NodeToReadable(object):
	"""Converts integer node ID's to human readable labels."""

	def __init__(self,
				 label_path=None,
				 uid_path=None):
		if not label_path:
			label_path = os.path.join(
				FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
		if not uid_path:
			uid_path = os.path.join(
				FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
		self.node_lookup = self.load(label_path, uid_path)

	def load(self, label_path, uid_path):
		"""Loads a human readable English name for each softmax node.
		Argumentss:
			label_path: string UID to integer node ID.
			uid_path: string UID to English string.
		Returns:
			dictionary from integer node ID to English string.
		"""
		if not tf.gfile.Exists(uid_path):
			tf.logging.fatal('File does not exist %s', uid_path)
		if not tf.gfile.Exists(label_path):
			tf.logging.fatal('File does not exist %s', label_path)

		# Loads mapping from string UID to English string
		proto_as_ascii_lines = tf.gfile.GFile(uid_path).readlines()
		uid_to_human = {}
		p = re.compile(r'[n\d]*[ \S,]*')
		for line in proto_as_ascii_lines:
			parsed_items = p.findall(line)
			uid = parsed_items[0]
			human_string = parsed_items[2]
			uid_to_human[uid] = human_string

		# Loads mapping from string UID to integer node ID.
		node_id_to_uid = {}
		proto_as_ascii = tf.gfile.GFile(label_path).readlines()
		for line in proto_as_ascii:
			if line.startswith('	target_class:'):
				target_class = int(line.split(': ')[1])
			if line.startswith('	target_class_string:'):
				target_class_string = line.split(': ')[1]
				node_id_to_uid[target_class] = target_class_string[1:-2]

		# Loads the final mapping of integer node ID to English string
		node_id_to_name = {}
		for key, val in node_id_to_uid.items():
			if val not in uid_to_human:
				tf.logging.fatal('Failed to locate: %s', val)
			name = uid_to_human[val]
			node_id_to_name[key] = name

		return node_id_to_name

	def id_to_string(self, node_id):
		if node_id not in self.node_lookup:
			return ''
		return self.node_lookup[node_id]


def create_graph():
	"""Creates a graph from saved GraphDef file and returns a saver."""
	# Creates graph from saved graph_def.pb.
	with tf.gfile.FastGFile(os.path.join(
			FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		_ = tf.import_graph_def(graph_def, name='')


def imageAnalyser(image_list, output_dir):
	"""Runs inference on an image list.
	Arguments:
		image_list: a list of images.
		output_dir: the directory in which image vectors will be saved
	Returns:
		image_to_labels: a dictionary with image file keys and predicted
			text label values
	"""
	image_to_labels = defaultdict(list)

	create_graph()

	with tf.Session() as sess:
		# Some useful tensors:
		# 'softmax:0': A tensor containing the normalized prediction across
		#	 1000 labels.
		# 'pool_3:0': A tensor containing the next-to-last layer containing 2048
		#	 float description of the image.
		# 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
		#	 encoding of the image.
		# Runs the softmax tensor by feeding the image_data as input to the graph.
		softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

		for image_index, image in enumerate(image_list):
			try:
				print("parsing", image_index, image, "\n")
				if not tf.gfile.Exists(image):
					tf.logging.fatal('File does not exist %s', image)
				
				with tf.gfile.FastGFile(image, 'rb') as f:
					image_data =	f.read()

					predictions = sess.run(softmax_tensor,
													{'DecodeJpeg/contents:0': image_data})

					predictions = np.squeeze(predictions)

					###
					# Get penultimate layer weights
					###

					feature_tensor = sess.graph.get_tensor_by_name('pool_3:0')
					feature_set = sess.run(feature_tensor,
													{'DecodeJpeg/contents:0': image_data})
					feature_vector = np.squeeze(feature_set)				
					outfile_name = os.path.basename(image) + ".npz"
					out_path = os.path.join(output_dir, outfile_name)
					np.savetxt(out_path, feature_vector, delimiter=',')

					# Creates node ID --> English string lookup.
					node_lookup = NodeToReadable()

					top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
					for node_id in top_k:
						human_string = node_lookup.id_to_string(node_id)
						score = predictions[node_id]
						print("results for", image)
						print('%s (score = %.5f)' % (human_string, score))
						print("\n")

						image_to_labels[image].append(
							{
								"labels": human_string,
								"score": str(score)
							}
						)

				# close the open file handlers
				proc = psutil.Process()
				open_files = proc.open_files()

				for open_file in open_files:
					file_handler = getattr(open_file, "fd")
					os.close(file_handler)
			except:
				print('could not process image index',image_index,'image', image)

	return image_to_labels


def downloadModel():
	"""Download and extract model tar file."""
	dest_directory = FLAGS.model_dir
	if not os.path.exists(dest_directory):
		os.makedirs(dest_directory)
	filename = DATA_URL.split('/')[-1]
	filepath = os.path.join(dest_directory, filename)
	if not os.path.exists(filepath):
		def _progress(count, block_size, total_size):
			sys.stdout.write('\r>> Downloading %s %.1f%%' % (
					filename, float(count * block_size) / float(total_size) * 100.0))
			sys.stdout.flush()
		filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
		print()
		statinfo = os.stat(filepath)
		print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
	tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
	downloadModel()
	if len(sys.argv) < 2:
		print("please provide folder where images are stored, e.g.")
		print("python classify_image_modified.py '../image_folder/*.jpg'")
		sys.exit()

	else:
		output_dir = "image_vectors"
		if not os.path.exists(output_dir):
			os.makedirs(output_dir)

		images = glob.glob(sys.argv[1])
		image_to_labels = imageAnalyser(images, output_dir)

		with open("image_to_labels.json", "w") as img_to_labels_out:
			json.dump(image_to_labels, img_to_labels_out)

		print("all done")
if __name__ == '__main__':
	tf.app.run()
'''
Results:
python check_duplicate_ranks.py "../duplicate/*"
parsing 01 ../duplicate.. 
.
.
.
parsing 107 ../duplicate/94_894415130.jpg 

results for ../duplicate/94_894415130.jpg
 (score = 0.88677)


results for ../duplicate/94_894415130.jpg
 (score = 0.03629)


results for ../duplicate/94_894415130.jpg
 (score = 0.00327)


results for ../duplicate/94_894415130.jpg
 (score = 0.00179)


results for ../duplicate/94_894415130.jpg
 (score = 0.00113)


results for ../duplicate/94_894415130.jpg
 (score = 0.00112)


results for ../duplicate/94_894415130.jpg
 (score = 0.00109)


parsing 108 ../duplicate/62_680613686.jpg 

results for ../duplicate/62_680613686.jpg
 (score = 0.80467)
.
.
.
.
all done
'''
