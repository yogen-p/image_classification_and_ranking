# Image_classification_ranking
Inception based model for Image classification and ranking

Image classification: The code below revolves around only a slight modification to this original script from TensorFlow’s ImageNet tutorial. The original script takes a single image as input and returns multiple string labels for the image as output. However we download the model, the script will print to the terminal several labels for the provided input image, each with a weight to show the model’s confidence for the given label. Then we obtain vector representations for image analyis basically instead of using the last (softmax) layer of the neural network for the text classifications of input images, we instead use the penultimate (second-last) layer of the neural network for the internal model weights for a given image, and store those weights as a vector representation of the input image. This will allow us to perform traditional vector analysis for these images. This method notes that the tensor pool_3:0 contains the weights for the penultimate layer of the network. These weights form a 2048 dimensional vector that’s perfect for image similarity computations.

Image similarity ranking: To measure similarity we use Annoy library from Spotify. The similar image viewer above uses ANN to identify similar images by craeting trees with 2048 dimension, the same we got from the penultimate layer vectors.

check_duplicate_ranks.py : Python code to generate vectors with Inception model labels for set of images 

vector_ranks.py : Python code to rank vectors produced by "check_duplicate_ranks.py" 

check_duplicate.py : Duplicate Image checker if two images are given 

dup_img_classifier.py : Duplicate image checker based on Inception. 
