#!/bin/bash

#Push some dir of files to Amazon S3

set -e

# Defaults -- I think this is how to do it.
# fromdir="."
# bucket="s3://nathan-caffe-snapshots"
# filter="*"

while [[ $# -gt 1 ]]
do
key="$1"
# Parse arguments, Override defaults if given
case $key in
	-b|--bucket)
	bucket="$2"
	shift
	;;
	-d|--fromdir)
	fromdir="$2"
	shift
	;;
	-f|--filter)
	filter="$2"
esac
shift
done

echo Files from DIR = "$fromdir"
echo To bucket = "$bucket"
echo Files to be copied:

files = $( ls $fromdir | wc -l )

if [ $files -gt 0 ]; then
# Do it
	aws s3 cp $fromdir $bucket --recursive --exclude * --include $filter
else
	echo "Skipping because $files files"
fi

# echo DONE
