#!/bin/bash

while read LINE
do
    arr=($LINE)
    qsub -V -cwd -m ae -M ing.nathany@gmail.com -o "out/${arr[3]}" -e "err/${arr[3]}" \
        -N ${arr[3]} ./crossval_run.sh ${arr[0]} ${arr[1]} ${arr[2]} ${arr[3]}
done < crossval_args.txt
