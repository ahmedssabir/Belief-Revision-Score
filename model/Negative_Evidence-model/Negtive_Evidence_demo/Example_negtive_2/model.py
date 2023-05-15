#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os


parser=argparse.ArgumentParser(description='call all scores and compute the visual context based Belief-revision')
parser.add_argument('--lm',  default='LM.txt', help='language model score (GPT2)', type=str,required=True)  
parser.add_argument('--sim', default='sim-score.txt', help='similarity score from fine_tune_BERT', type=str,required=False)  
parser.add_argument('--vis', default='visual-context_label.txt',help='class-label from the classifier (Resent152)', type=str, required=True)  
parser.add_argument('--vis_prob', default='visual-context.txt', help='prob from the classifier (Resent152)', type=str, required=True) 
parser.add_argument('--c',  default='caption.txt', help='caption from the baseline (any)', type=str, required=True) 
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


# hypothesis revision based visual re-ranker
def visual_reranker_negtive_visual(LM, visual_context_init, sim):
     
    	#score = pow(float(LM),pow((1-float(sim))/(1+ float(sim)),1-float(classifier_confidence))) 
        score = 1 - pow((1-float(LM)),pow((1-float(sim))/(1+ float(sim)),1-float(visual_context_init)))      
  
    	return score

# remove duplicate beam and re-rank top-k caption after the revision 
def remove_duplicate_caption_re_rank(input_path, output_path):

    with open(input_path, 'r') as input_file, open(output_path, 'w') as output_file:
        seen_lines = set()

        def add_line(line):
            seen_lines.add(line)
            return line

        output_file.writelines((add_line(line) for line in input_file
                                if line not in seen_lines))
    re_ranked_scores = []
    with open(output_path) as f:
          for line in f:
              caption, score = line.split(',')
              score = float(score)
              re_ranked_scores.append((caption, score))
    re_ranked_scores.sort(key=lambda s: float(s[1]), reverse=True)
    # re-ranked caption     
    with open(output_path, 'w') as f:
    	for caption, score in re_ranked_scores:
           f.write("%s %s\n" % (caption, score))
    	   #print(caption, score)

    
           
# all beam with visual context 
input_path= 'Belief-revision.txt'
# re-ranked beam with visual context 
output_path = 'Belief-revision_re-rank.txt'

# compute visual context
f=open(input_path, "w")
for i in range(len(get_lines(args.lm))):
    temp =[]
    LM  = get_lines(args.lm)[i]
    #sim = get_lines('sim-score.txt')[i]
    visual_context_label = get_lines(args.visNg)[i]
    visual_context_init = get_lines(args.visNg_init)[i]
    caption = get_lines(args.c)[i]

   
    caption_emb = model.encode(caption, convert_to_tensor=True)
    visual_context_label_emb = model.encode(visual_context_label, convert_to_tensor=True)

    #def cos_sim(a, b):
    #    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


    
    sim =  cosine_scores = util.pytorch_cos_sim(caption_emb, visual_context_label_emb)
    sim = sim.cpu().numpy()
    sim = sim.item()



    # model 
    #score = visual_reranker(LM, visual_context_prob, sim)
    score = visual_reranker_negtive_visual(LM, visual_context_init, sim)
    #print(score)
    temp.append(score)
    
 
    
    #result = ','.join((caption, LM, str(score))) 
    result = ','.join((caption, str(score))) 
    result = re.sub(r'\s*,\s*', ',', result) 


    #print(result)     

    f.write(result)
    f.write('\n')

        
f.close()

# re-rank and print top captions
remove_duplicate_caption_re_rank(input_path, output_path)	






