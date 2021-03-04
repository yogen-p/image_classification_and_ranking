# Duplicate image checker based on Inception. 

from __future__ import absolute_import, division, print_function

import os
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
from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, os, codecs, random
import numpy as np
import argparse
FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   Binary representation of the GraphDef protocol buffer.
# imagenet_synset_to_human_label_map.txt:
#   Map from synset ID to a human readable string.
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   Text representation of a protocol buffer mapping a label to synset ID.
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class NodeLookup(object):
  """Converts integer node ID's to human readable labels."""

  def __init__(self,
               label_lookup_path=None,
               uid_lookup_path=None):
    if not label_lookup_path:
      label_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
    if not uid_lookup_path:
      uid_lookup_path = os.path.join(
          FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
    self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

  def load(self, label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.
    Returns:
      dict from integer node ID to human-readable string.
    """
    if not tf.gfile.Exists(uid_lookup_path):
      tf.logging.fatal('File does not exist %s', uid_lookup_path)
    if not tf.gfile.Exists(label_lookup_path):
      tf.logging.fatal('File does not exist %s', label_lookup_path)

    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')
    for line in proto_as_ascii_lines:
      parsed_items = p.findall(line)
      uid = parsed_items[0]
      human_string = parsed_items[2]
      uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
      if line.startswith('  target_class:'):
        target_class = int(line.split(': ')[1])
      if line.startswith('  target_class_string:'):
        target_class_string = line.split(': ')[1]
        node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
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


def run_inference_on_images(image, output_dir):
  """Runs inference on an image list.

  Args:
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
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')

    # for image_index, image in enumerate(image_list):
    try:
      # print("parsing", image_index, image, "\n")
      if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
      
      with tf.gfile.FastGFile(image, 'rb') as f:
        image_data =  f.read()

        predictions = sess.run(softmax_tensor,
                        {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)
        print(len(predictions))
        print("NumPy Squeezing Part 1 Done.")
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
        print("NumPy Squeezing Part 2 Done.")
        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

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
    except Exception as e:
      print(e)
      print('could not process image index')#,image_index,'image', image

  return image_to_labels


def maybe_download_and_extract():
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
  maybe_download_and_extract()
  if len(sys.argv) < 2:
    print("please provide a glob path to one or more images, e.g.")
    print("python classify_image_modified.py '../cats/*.jpg'")
    sys.exit()

  else:
    output_dir = "results"
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    
    image1 = sys.argv[1]
    image2 = sys.argv[2]
    image1_file_name = sys.argv[1].split("/")[-1]
    image2_file_name = sys.argv[2].split("/")[-1]
    master_image1_labels = run_inference_on_images(image1, output_dir)
    master_image2_labels = run_inference_on_images(image2, output_dir)
    final_image_labels_dict = {**master_image1_labels, **master_image2_labels}
    with open("image1_to_labels.json", "w") as master_image_labels_out:
      json.dump(final_image_labels_dict, master_image_labels_out)
    print ("Dumping done.")


  # data structures
  file_index_to_file_name = {}
  file_index_to_file_vector = {}

  # config
  dims = 2048
  n_nearest_neighbors = 1
  trees = 10000
  infiles = glob.glob('results/*.npz')

  # build ann index
  t = AnnoyIndex(dims)
  for file_index, i in enumerate(infiles):
    file_vector = np.loadtxt(i)
    file_name = os.path.basename(i).split('.')[0]
    file_index_to_file_name[file_index] = file_name
    file_index_to_file_vector[file_index] = file_vector
    t.add_item(file_index, file_vector)
  t.build(trees)

  # create a nearest neighbors json file for each input
  if not os.path.exists('nearest_neighbors'):
    os.makedirs('nearest_neighbors')
  named_nearest_neighbors = []
  image1_vectors = {}
  image2_vectors = {}
  image1_vectors['name'] = file_index_to_file_name[0]
  image1_vectors['vectors'] = file_index_to_file_vector[0]
  image2_vectors['name'] = file_index_to_file_name[1]
  image2_vectors['vectors'] = file_index_to_file_vector[1]

  similarity = 1 - spatial.distance.cosine(image1_vectors['vectors'], image2_vectors['vectors'])
  rounded_similarity = (int((similarity * 10000)) / 10000.0)*100
  print(rounded_similarity)
  if rounded_similarity > 80.00:
    print ("Duplicate")
  else:
    print ("Distinct")
  print("all done")
  os.system("rm -rf results/*.npz")
if __name__ == '__main__':
  tf.app.run()
