#!/bin/bash

set -e
qsub -V -cwd -N "nuclei-0-choose20" ./matlabjob0.sh 
qsub -V -cwd -N "nuclei-1-choose20" ./matlabjob1.sh 
exit 0
