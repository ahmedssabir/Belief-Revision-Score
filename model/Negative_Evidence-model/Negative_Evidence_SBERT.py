#!/usr/bin/env python3
import sys
import argparse
import torch
import re

# how to run
# python negative_evidence_SBERT.py --lm LM.txt --visNg negative_visual.txt  --visNg_init negative_visual_init.txt --c caption.txt

parser=argparse.ArgumentParser(description='call all scores and compute the Negative_Evidence based visual context re-ranker')
parser.add_argument('--lm',  default='LM.txt', help='language model score (GPT2)', type=str,required=False)  
parser.add_argument('--sim', default='sim-score.txt', help='similarity score from fine_tune_BERT', type=str,required=False)  
parser.add_argument('--visNg', default='Negtive_visual.txt',help='Ng visual context, similar vector (GloVe) to the original class label(Resent152)', type=str, required=False)  
parser.add_argument('--visNg_init', default='Negtive_visual_init.txt', help='init visNg with language model', type=str, required=False) 
parser.add_argument('--c',  default='caption.txt', help='caption from the baseline (any)', type=str, required=False) 
args = parser.parse_args()


from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
#model = SentenceTransformer('nq-distilbert-base-v1')


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')

resutl=[]

# output
f=open('Belief-revision-with-Negative_Evidence.txt', "w")
for i in range(len(get_lines(args.lm))):
    temp =[]
    LM  = get_lines(args.lm)[i]
    #sim = get_lines('sim-score.txt')[i]
    visual_context_label = get_lines(args.visNg)[i]
    visual_context_init = get_lines(args.visNg_init)[i]
    caption = get_lines(args.c)[i]

   
    embeddings1 = model.encode(caption, convert_to_tensor=True)
    embeddings2 = model.encode(visual_context_label, convert_to_tensor=True)

    #def cos_sim(a, b):
    #    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


   
    sim =  cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    sim = sim.cpu().numpy()
    sim = str(sim)[1:-1]
    sim = str(sim)[1:-1]   


    score = 1 - pow((1-float(LM)),pow((1-float(sim))/(1+ float(sim)),1-float(visual_context_init)))
    #print(score)
    temp.append(score)

    #result= caption +','+ LM +','+ str(score)
    result = ','.join((caption, LM, str(score))) 
    result = re.sub(r'\s*,\s*', ',', result)   
    print(result)     

    f.write(result)
    f.write('\n')
    

    
f.close()





