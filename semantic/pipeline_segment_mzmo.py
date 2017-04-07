#!/usr/bin/python

'''
/home/nathan/semantic-ccrcc/code/pipeline_segment_wsi.py
Apply Pipeline:
	1. Tile svs

	2. Apply model to each tile

	3. Write out reassembled wsi

Arguments, by step:
	1. svs_path, write_path
	2. model_path, write_path
	3. svs_path, write_path

Outputs, by step: (intermediates & write-outs)
	1. tiles, list.txt, data_tilemap.npy
	2. classified tiles, statistics.npy (TODO)
	3. _wsi_colored.jpg

'''
import segment_mzmo
import replace_text
import os
import shutil
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--write_home', type=str, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
# parser.add_argument('--cleanup', action="store_true")

args = parser.parse_args()

path   = os.path.expanduser(args.path)
write_home   = os.path.expanduser(args.write_home)
model = os.path.expanduser(args.model)
weights = os.path.expanduser(args.weights)


listfile = os.path.join(path, 'list.txt')
logfile = os.path.join(write_home, 'process_log.txt')

inference_file = os.path.join(write_home, 'segnet_basic_inference.prototxt') # Write into
replace_text.run(filein=model, #Read from
			     replacement=listfile, #To be placed
			     target_text="PLACEHOLDER", #Replace this text
			     filewrite=inference_file) #Put the new file here


rgb_dir = os.path.join(write_home, "rgb")
labels_dir = os.path.join(write_home, "labels")
prob_dir = os.path.join(write_home, "probability")
segment_mzmo.run(model = inference_file, 
				  weights = weights, 
				  tiledir = path, 
				  write_rgb = rgb_dir,
				  write_labels = labels_dir,
				  write_prob = prob_dir,
				  cleanup = True)

# ##3. 
# img_dir = os.path.join(dir_root, 'tiles')
# # Still needs work
# assemble_tiles.run(source_rgb = rgb_dir,
# 				   source_labels = labels_dir,
# 				   source_prob = prob_dir,
# 			       writeto = dir_root,
# 			       monitor = False,
# 			       imgpath = img_dir)

# ##4. Clean
# if args.cleanup:
# 	shutil.rmtree(img_dir)
# 	shutil.rmtree(rgb_dir)
# 	shutil.rmtree(labels_dir)
# 	shutil.rmtree(prob_dir)

