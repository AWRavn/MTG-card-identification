#!pip uninstall opencv-python -y
#!pip install opencv-contrib-python==3.4.2.17 --force-reinstall

import os
from PIL import Image
import cv2
import numpy as np
import time
import argparse

def load_images(DIRECTORY_PATH):
	"""
	Produces a dictionary of images from a given directory.

	Args:
		DIRECTORY_PATH (str):					Path to the directory containing target images.

	Returns:
		img_dict_list (list(dict)):				Dictionary of cards/photos.
			name (str):							Name of the card/photograph.
			image_GRAY (cv2.COLOR_BGR2GRAY):	Grayscale image of the card/photograph.

	Raises:
		FileNotFoundError:						If the directory doesn't exist.

	"""

	img_dict_list = []

	try:
		for filename in os.listdir(DIRECTORY_PATH):
			img = cv2.imread(DIRECTORY_PATH + filename)
			if type(img) is np.ndarray:
				img_GRAY = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
				img_dict = {
					'name': os.path.splitext(filename)[0],
					'image_GRAY': img_GRAY,
				}
				img_dict_list.append(img_dict)
			else:
				continue
	except FileNotFoundError:
		raise

	return img_dict_list

	
def get_features(img_dict_list, model):
	"""
	Supplements the card dictionary with keypoints and descriptors from the chosen model. Available models:
	ORB, SIFT.

	Args:
		img_dict_list (list(dict)):				Dictionary of cards/photos.
			name (str):							Name of the card/photograph.
			image_GRAY (cv2.COLOR_BGR2GRAY):	Grayscale image of the card/photograph.
	model (str):								'SIFT' or 'ORB' feature matching model.

	Returns:
		img_dict_list (list(dict)):				Dictionary of cards/photos.
			name (str):							Name of the card/photograph.
			image_GRAY (cv2.COLOR_BGR2GRAY):	Grayscale image of the card/photograph.
			keypoint (keypoints):				Keypoints of the card/photograph.
			descriptor (descriptors):			Descriptors of the card/photograph.
	
	Raises:
		ArgumentTypeError:						If an unimplemented model is selected. Or input is not in a 
												correct dictionary format.

	"""

	# Check input arguments
	if model == 'ORB':
		model = cv2.ORB_create()
	elif model == 'SIFT':
		model = cv2.SIFT_create()
	else:
		raise argparse.ArgumentTypeError('Unsupported model. Use "SIFT" or "ORB".')

	if not 'name' in img_dict_list[0].keys() and 'image_GRAY' in img_dict_list[0].keys():
		raise argparse.ArgumentTypeError('Input must be a dictionary with keys "name" and "image_GRAY".')


	for img_dict in img_dict_list:
		keypoint, descriptor = model.detectAndCompute(img_dict['image_GRAY'], None)
		img_dict['keypoint'] = keypoint
		img_dict['descriptor'] = descriptor

	return img_dict_list

def get_matches(source_dict_list, target_dict_list, model='SIFT', write=True):
	"""
	Produces a report of matching accuracy between targer and source dictionaries containing descriptors.
	Assumes photo names start with set code and end with the correct card name.

	Args:
		source_dict_list (list(dict)):			Dictionary of photographed cards.
			name (str):							Name of the photograph.
			image_GRAY (cv2.COLOR_BGR2GRAY):	Grayscale image of the photograph.
			keypoint (keypoints):				Keypoints of the photograph.
			descriptor (descriptors):			Descriptors of the photograph.
		target_dict_list (list(dict)):			Dictionary of the target cards.
			name (str):							Name of thet target card.
			image_GRAY (cv2.COLOR_BGR2GRAY):	Grayscale image of the target card.
			keypoint (keypoints):				Keypoints of the target card.
			descriptor (descriptors):			Descriptor of the target card.
		model (str):							'SIFT' or 'ORB' feature matching model.
		write (bool):							True if write a report to file. Default true.

	Returns:
		corr_list (list [str]):					List of the photos correctly matched.
		par_err_list (list [str]):				List of the photos partially matched.
		err_list (list [str]):					List of the photos mismatched.

	Raises:
		ArgumentTypeError:						If an unimplemented model is selected. Or input is not in a 
												correct dictionary format.

	"""

	# Check input arguments
	if model == 'ORB':
		bfm = cv2.BFMatcher(cv2.NORM_HAMMING)
	elif model == 'SIFT':
		bfm = cv2.BFMatcher()
	else:
		raise argparse.ArgumentTypeError('Unsupported model. Use "SIFT" or "ORB".')

	if not 'name' in source_dict_list[0].keys() and 'image_GRAY' in source_dict_list[0].keys() and \
	'keypoint' in source_dict_list[0].keys() and 'descriptor' in source_dict_list[0].keys():
		raise argparse.ArgumentTypeError('Source input must be a dictionary with keys "name" and "image_GRAY".')

	if not 'name' in target_dict_list[0].keys() and 'image_GRAY' in target_dict_list[0].keys() and \
	'keypoint' in target_dict_list[0].keys() and 'descriptor' in target_dict_list[0].keys():
		raise argparse.ArgumentTypeError('Source input must be a dictionary with keys "name" and "image_GRAY".')

	# Accuracy counters and error lists
	c_corr = 0
	c_par = 0
	tot = len(source_dict_list)
	err_list = []
	par_err_list = []
	corr_list = []

	for source_dict in source_dict_list:
		
		match_names = []
		match_counts = []

		for target_dict in target_dict_list:
			matches = bfm.knnMatch(source_dict['descriptor'], target_dict['descriptor'], k=2)

			# Apply ratio test
			good_matches = []
			for m, n in matches:
				if m.distance < 0.75*n.distance:
					good_matches.append([m])

			match_names.append(target_dict['name'])
			match_counts.append(len(good_matches))

		# Get best match
		best_idx = np.argmax(match_counts)
		#print('Matched ', source_dict['name'], ' to: ', match_names[best_idx], ", with ", match_counts[best_idx], " matches") 

		# Accuracy tracker
		if match_names[best_idx][-5:]==source_dict['name'][-5:]:
			c_par = c_par + 1
			if match_names[best_idx][:4]==source_dict['name'][:4]:
				c_corr = c_corr + 1
				corr_list.append(source_dict['name'])
			else:
				par_err_list.append(source_dict['name'])
		else:
			err_list.append(source_dict['name'])

	# Write to a file
	if write == True:	
		
		# Create directory if needed
		os.makedirs('../reports', exist_ok=True)

		with open(f'../reports/{time.strftime("%Y%m%d %H-%M-%s")}.txt', 'w') as out_file:
			out_file.write(f'Completely correct cards: {c_corr} / {tot}\n')
			out_file.write(f'Partially correct cards: {c_par-c_corr} / {tot}\n')
			out_file.write(f'Misclassified cards: {tot-c_par} / {tot}\n')
			out_file.write('Cards classified correctly:\n')
			for card in corr_list:
				if card == corr_list[-1]:
					out_file.write(f'{card}.\n')
				else:
					out_file.write(f'{card}, ')
			out_file.write('Cards with correct name but wrong edition:\n')
			for card in par_err_list:
				if card == par_err_list[-1]:
					out_file.write(f'{card}.\n')
				else:
					out_file.write(f'{card}, ')
			out_file.write('Cards misclassified:\n')
			for card in err_list:
				if card == err_list[-1]:
					out_file.write(f'{card}.\n')
				else:
					out_file.write(f'{card}, ')

	return corr_list, par_err_list, err_list

def main():
	"""
	Example usage.
	"""

	print('Loading data')

	# Get paths
	PATH_TO_SOURCE = "../images/photos/"
	PATH_TO_TARGET = "../images/database/"
	PATH_TO_CROP = "../images/art-crop/"

	source_path_list = load_images(PATH_TO_SOURCE) 
	target_path_list = load_images(PATH_TO_TARGET) 
	crop_path_list = load_images(PATH_TO_CROP)

	write = True

	# Full art
	print('Full art data')

	# Run ORB
	print('Begin matching ORB')
	model = 'ORB'
	source_ORB = get_features(source_path_list, model)
	target_ORB = get_features(target_path_list, model)
	corr_list_1, par_err_list_1, err_list_1 = get_matches(source_ORB, target_ORB, model, write)

	# Run SIFT
	print('Begin matching SIFT')
	model = 'SIFT'
	source_SIFT = get_features(source_path_list, model)
	target_SIFT = get_features(target_path_list, model)
	corr_list_2, par_err_list_2, err_list_2 = get_matches(source_SIFT, target_SIFT, model, write)
	

	# Art crop
	print('Cropped data')

	# Run ORB
	print('Begin matching ORB')
	model = 'ORB'
	source_ORB = get_features(source_path_list, model)
	crop_ORB = get_features(crop_path_list, model)
	corr_list_3, par_err_list_3, err_list_3 = get_matches(source_ORB, crop_ORB, model, write)

	# Run SIFT
	print('Begin matching SIFT')
	model = 'SIFT'
	source_SIFT = get_features(source_path_list, model)
	crop_SIFT = get_features(crop_path_list, model)
	corr_list_4, par_err_list_4, err_list_4 = get_matches(source_SIFT, crop_SIFT, model, write)


if __name__ == "__main__":
	main()