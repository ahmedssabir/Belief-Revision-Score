#!/bin/sh
python SDiv_SBERT_eval.py --hyp k-5.txt

 awk -F';' '{sum+=$1; ++n} END { print "Avg: "sum"/"n"="sum/n }' < score.txt
