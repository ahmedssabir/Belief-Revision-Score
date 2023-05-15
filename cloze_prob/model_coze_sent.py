#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os


# How to run

#python model_coze.py --c caption.txt --vis_prob visual_context_prob.txt  --vis visual_context_label.txt

parser = argparse.ArgumentParser(description='call all scores and compute the visual context based Belief-revision')
#parser.add_argument('--lm', default='LM.txt', help='language model score (GPT2)', type=str, required=False)
#parser.add_argument('--sim', default='sim-score.txt', help='similarity score from fine_tune_BERT', type=str, required=False)
#parser.add_argument('--vis', default='visual-context_label.txt', help='class-label from the classifier (Resent152)', type=str, required=True)
parser.add_argument('--sent', default='set.txt', help='caption from the baseline (any)', type=str, required=True)
parser.add_argument('--sent_context', default='visual-context.txt', help='prob from the classifier (Resent152)', type=str,required=True)
parser.add_argument('--output', default='', help='output', type=str, required=True)


args = parser.parse_args()

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from get_probabilities import cloze_prob 

model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
        return f.read().strip().split('\n')


# hypothesis revision based cloze prob 
class Visual_re_ranker:
    def __init__(self, sent_LM, sent_context_LM, sim):
        self.sent_LM = sent_LM
        self.sent_context_LM = sent_context_LM
        self.sim = sim

    def belief_revision(self):
        score = pow(float(sent_LM), pow((1 - float(sim)) / (1 + float(sim)), 1 - float(sent_context_LM)))
        return score




output_path = args.output 

# compute visual context
f = open(output_path, "w")
for i in range(len(get_lines(args.sent))):
    temp = []
    sent = get_lines(args.sent)[i]
    sent_context = get_lines(args.sent_context)[i]
 

    sent_LM = cloze_prob(sent)
    sent_context_LM = cloze_prob(sent_context)    

    sent_emb = model.encode(sent, convert_to_tensor=True)
    sent_context_emb = model.encode(sent_context, convert_to_tensor=True)

    sim = cosine_scores = util.pytorch_cos_sim(sent_emb, sent_context_emb)
    sim = sim.cpu().numpy()
    sim = sim.item()
    

    score = Visual_re_ranker(sent_LM, sent_context_LM, sim)
    score = score.belief_revision()
    temp.append(score)
    
    print("{} {} {}".format('colze_prob_sent',sent, sent_LM))
    print("{} {} {}".format('colze_prob_1',sent_context_LM, score))
    print("{} {} {}".format('sim', sent, sim))
    print("{} {} {}".format('score', sent, score))

    # result = ','.join((caption, LM, str(score)))
    result = ','.join((sent, str(score)))
    result = re.sub(r'\s*,\s*', ',', result)


    f.write(result)
    f.write('\n')

f.close()

