#  Python code to rank vectors produced by "check_duplicate_ranks.py" 
from annoy import AnnoyIndex
from scipy import spatial
from nltk import ngrams
import random, json, glob, os, codecs, random
import numpy as np
import sys
# data structures
file_index_to_file_name = {}
file_index_to_file_vector = {}
chart_image_positions = {}

# config
dims = 2048
n_nearest_neighbors = 7
trees = 10000
infiles = glob.glob('image_vectors/*.npz')

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

for i in file_index_to_file_name.keys():
	master_file_name = file_index_to_file_name[i]
	master_vector = file_index_to_file_vector[i]

	named_nearest_neighbors = []
	nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)
	for j in nearest_neighbors:
		neighbor_file_name = file_index_to_file_name[j]
		neighbor_file_vector = file_index_to_file_vector[j]

		similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
		rounded_similarity = int((similarity * 10000)) / 10000.0

		named_nearest_neighbors.append({
			'filename': neighbor_file_name,
			'similarity': rounded_similarity
		})

	with open('nearest_neighbors/' + master_file_name + '.json', 'w') as out:
		json.dump(named_nearest_neighbors, out)


def main():
	if len(sys.argv) < 2:
		print("please provide an image")
		print("python classify_image_modified.py 'findSimilar_to_image.jpg'")
		sys.exit()
	else:
		image = sys.argv[1]
		file_path = os.path.splitext(image)
		image_name = file_path[0].split('/')[-1]
		with open('nearest_neighbors/'+ image_name + '.json', "r") as image_json:
			data = json.load(image_json)
			print(json.dumps(data, indent=4))
if __name__ == '__main__':
	main()
	
'''
Results:
python vector_ranks.py /home/nv/Documents/test-run/duplicate/25_758351074.jpg
[
    {
        "filename": "25_758351074",
        "similarity": 1.0
    },
    {
        "filename": "51_8126078238",
        "similarity": 0.9077
    },
    {
        "filename": "25_6843863532",
        "similarity": 0.9077
    },
    {
        "filename": "82_9428376609",
        "similarity": 0.9063
    },
    {
        "filename": "26_2614960844",
        "similarity": 0.9048
    },
    {
        "filename": "88_7139005155",
        "similarity": 0.9017
    },
    {
        "filename": "17_828296168",
        "similarity": 0.8571
    }
]
'''
