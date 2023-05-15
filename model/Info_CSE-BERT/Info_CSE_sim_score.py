#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os
import torch
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer

#paper InfoCSE: Information-aggregated Contrastive Learning of Sentence Embeddings 10-2022
#https://arxiv.org/pdf/2210.06432.pdf

# Import the models. The package will take care of downloading the models automatically
#tokenizer = AutoTokenizer.from_pretrained("ffgcc/gsInfoNCE-roberta-large")
#model = AutoModel.from_pretrained("ffgcc/gsInfoNCE-roberta-large")

tokenizer = AutoTokenizer.from_pretrained("ffgcc/InfoCSE-bert-base")
model = AutoModel.from_pretrained("ffgcc/InfoCSE-bert-base")

parser=argparse.ArgumentParser(description='compute similarity score between caption and visual context')
parser.add_argument('--c',  default='caption.txt', help='beam serach', type=str, required=True)
parser.add_argument('--vis', default='',help='visual context', type=str, required=True)
parser.add_argument('--output', default='ouput_file', help='result file', type=str, required=True)

args = parser.parse_args()


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')
    
           
# output path
input_path= args.output 

# compute visual context
f=open(input_path, "w")
for i in range(len(get_lines(args.vis))):
    temp =[]
    visual_context_label = get_lines(args.vis)[i]
    caption = get_lines(args.c)[i]

    # Tokenize input texts
    embeddings1= tokenizer(visual_context_label, padding=True, truncation=True, return_tensors="pt")
    embeddings2 = tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
       
    #embeddings1 = tokenizer(caption, padding=True, truncation=True, return_tensors="pt")
    #embeddings2 = tokenizer(visual_context_label, padding=True, truncation=True, return_tensors="pt")
    
    # get the embedding 
    with torch.no_grad():
        #embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
	    caption_embedding = model(**embeddings1, output_hidden_states=True, return_dict=True).pooler_output
	    visual_context_label_embedding = model(**embeddings2, output_hidden_states=True, return_dict=True).pooler_output

    sim =  cosine_sim_0_1 = 1 - cosine(caption_embedding[0],  visual_context_label_embedding[0])
   

    temp.append(sim)
    
    result = ','.join((caption,  visual_context_label, str(sim)))
    #print(result)

    f.write(result)
    f.write('\n')

        
f.close()
