#!/bin/sh
python MTLD.py --hyp beam.txt  --output score.txt 

 awk -F';' '{sum+=$1; ++n} END { print "Avg: "sum"/"n"="sum/n }' < score.txt
