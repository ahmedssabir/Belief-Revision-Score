#!/usr/bin/env python3
import argparse
import re
import torch 
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import statistics


parser=argparse.ArgumentParser()
parser.add_argument('--ref_1',  default='karpathy-refs-1.txt', help='human annotation', type=str,required=False)  
parser.add_argument('--ref_2',  default='karpathy-refs-2.txt', help='human annotation', type=str,required=False)  
parser.add_argument('--ref_3',  default='karpathy-refs-3.txt', help='human annotation', type=str,required=False)  
parser.add_argument('--ref_4',  default='karpathy-refs-4.txt', help='human annotation', type=str,required=False) 
parser.add_argument('--ref_5',  default='karpathy-refs-5.txt', help='human annotation', type=str,required=False) 
parser.add_argument('--hyp', default='beam_search.txt', help='caption baseline', type=str,required=False)  
args = parser.parse_args()


model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')
  
def compute_sim(ref,hyp):
     ref_embed = model.encode(ref, convert_to_tensor=True)
     hyp_embed = model.encode(hyp, convert_to_tensor=True)
     sim = cosine_scores = util.pytorch_cos_sim(ref_embed, hyp_embed)
     sim.item()
     #temp.append(sim)
     return  sim

def MaxValue():
    max_value = max(sim_result)
    return max_value

output_path= 'cosine_score_baseline_b5.txt'
f=open(output_path, "w")
for i in range(len(get_lines(args.ref_1))):
    temp =[]
    ref_1 = get_lines(args.ref_1)[i]
    ref_2 = get_lines(args.ref_2)[i]
    ref_3 = get_lines(args.ref_3)[i]
    ref_4 = get_lines(args.ref_4)[i]
    ref_5 = get_lines(args.ref_5)[i]
    hyp = get_lines(args.hyp)[i]
   
    ref_1_sim = compute_sim(ref_1 ,hyp)
    ref_2_sim = compute_sim(ref_2, hyp)
    ref_3_sim = compute_sim(ref_3,hyp)
    ref_4_sim = compute_sim(ref_4 ,hyp)
    ref_5_sim = compute_sim(ref_5,hyp)

    sim_result = [ref_1_sim.item(), ref_2_sim.item(),ref_3_sim.item(),ref_4_sim.item(),ref_5_sim.item()]
    Max_score = MaxValue()

    result = str(MaxValue())
    #result = ','.join((ref, hyp, str(ref_1_sim), ref_2_sim, ref_3_sim, ref_4_sim, ref_5_sim))
    result = re.sub(r'\s*,\s*', ',', result) 

    #statistics.mean(l) 

    f.write(result)
    f.write('\n')
    print(result)

f.close()

