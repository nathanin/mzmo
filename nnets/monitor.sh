# Run a script and periodically push some folder to S3

# monitor_dir="/Users/nathaning/Dropbox/projects/S3/testdir"

monitor_dir="/home/ubuntu/mzmo/models"
bucket_models="s3://nathan-caffe-snapshots"
bucket_logs="s3://nathan-caffe-logs"

counter=0
while [ $counter -lt 10 ]; do
	date
	ls $monitor_dir
# DO it!
	./pushS3.sh \
	--bucket $bucket \
	--fromdir $monitor_dir \
	--filter "*.caffemodel"

	echo Pushed model snapshots:
	files=$( ls $monitor_dir/*.caffemodel | wc -l )

	if [ $files -gt 0 ]; then
		ls $monitor_dir/*.caffemodel
	else
		echo "Pushed nothing"
	fi

# DO it!
	./pushS3.sh \
	--bucket $bucket \
	--fromdir $monitor_dir \
	--filter "caffe.INFO"

	echo Pushed LOG:
	ls $monitor_dir/*.caffemodel

	echo Waiting ...
	sleep 1h
	# echo 'OK!' > $monitor_dir/$counter.txt
	# let counter=counter+1

done


# for i in $( ls $monitor_dir ); do
# 	echo $monitor_dir/$i
# 	echo contains:
# 	cat $monitor_dir/$i

# done
