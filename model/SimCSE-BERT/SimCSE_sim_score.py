#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

#2021 model 
# https://www.sbert.net/docs/pretrained_models.html
#model = SentenceTransformer('paraphrase-mpnet-base-v2')

# 
#python SBERT_sim_score.py  --caption caption.txt --vis visual_context_label.txt


parser=argparse.ArgumentParser()
parser.add_argument('--caption',  default='caption.txt', help='beam serach', type=str,required=True)  
parser.add_argument('--vis', default='visual_context_label.txt', help='visual_context from ResNet', type=str,required=True)  
args = parser.parse_args()


# from SimCSE paper 
# Import our models. The package will take care of downloading the models automatically
tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')

 
output_path = 'SimCSE_caption-visual_context_score.txt'

# compute visual context
f=open(output_path, "w")
for i in range(len(get_lines(args.caption))):
    temp =[]
    caption  = get_lines(args.caption)[i]
    visual_context_label = get_lines(args.vis)[i]

    caption_emb = tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
    visual_context_label_emb = tokenizer(visual_context_label, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
	    caption_embedding = model(**caption_emb, output_hidden_states=True, return_dict=True).pooler_output
	    visual_context_label_embedding = model(**visual_context_label_emb, output_hidden_states=True, return_dict=True).pooler_output
   
    sim =  cosine_sim_0_1 = 1 - cosine(caption_embedding[0],  visual_context_label_embedding[0])
    temp.append(sim)

    # to print visual-context,caption,score
    #result= file1[i]+','+file2[i]+','+str(w)
    
    # score only 
    result= str(sim)
    
    f.write(result)
    f.write('\n')
    
    
f.close()
