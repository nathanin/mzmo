#!/usr/bin/python

import argparse

def run(filein, replacement, target_text, filewrite):
	f = open(filein, "r")
	filedata = f.read()
	f.close()

	newdata = filedata.replace(target_text,replacement)

	f = open(filewrite, "w")
	f.write(newdata)
	f.close()

if __name__=='__main__':
	p = argparse.ArgumentParser()
	p.add_argument('filein')
	p.add_argument('target_text')
	p.add_argument('replacement')
	p.add_argument('filewrite')
	args = p.parse_args()

	filein = args.filein
	replacement = args.replacement
	target_text = args.target_text
	filewrite = args.filewrite

	run(filein, replacement, target_text, filewrite)
