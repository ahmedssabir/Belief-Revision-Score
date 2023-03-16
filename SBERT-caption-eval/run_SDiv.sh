#!/bin/sh
python SDiv_SBERT_eval.py  --ref_1 ref_karpath/karpathy-refs-1.txt  --ref_2 ref_karpath/karpathy-refs-2.txt --ref_3 ref_karpath/karpathy-refs-3.txt --ref_4 ref_karpath/karpathy-refs-4.txt  --ref_5 ref_karpath/karpathy-refs-5.txt  --hyp k3_BLIP_hyp.txt --output score.txt

 awk -F';' '{sum+=$1; ++n} END { print "Avg: "sum"/"n"="sum/n }' < score.txt

