#!/usr/bin/env python3
import sys
import argparse
import torch
import re
import os


# How to run
# python model.py --sent sent.txt --context_sent context_sent.txt --output output.txt


parser = argparse.ArgumentParser(description='call all scores and compute the visual context based Belief-revision')
#parser.add_argument('--lm', default='LM.txt', help='language model score (GPT2)', type=str, required=False)
#parser.add_argument('--sim', default='sim-score.txt', help='similarity score from fine_tune_BERT', type=str, required=False)
parser.add_argument('--sent', default='sentence.txt', help='', type=str, required=True)
parser.add_argument('--context_sent', default='context_sent.txt', help='', type=str, required=True)
parser.add_argument('--output', default='', help='output', type=str, required=True)
args = parser.parse_args()

from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from lm_scorer.models.auto import AutoLMScorer as LMScorer


# GPT2-large Load model to cpu or cuda 
device = "cuda:0" if torch.cuda.is_available() else "cpu"
batch_size = 1
scorer = LMScorer.from_pretrained("gpt2-large", device=device, batch_size=batch_size)
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')

def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
        return f.read().strip().split('\n')


# hypothesis revision based sentence similarity
class Visual_re_ranker:
    def __init__(self, sent, context_sent, sim):
        self.sent = sent
        self.context_sent = context_sent
        self.sim = sim

    def belief_revision(self):
        score = pow(float(sent), pow((1 - float(sim)) / (1 + float(sim)), 1 - float(context_sent)))
        return score


#input_path = 'Belief-revision.txt'
input_path = args.output


f = open(input_path, "w")
for i in range(len(get_lines(args.sent))):
    temp = []
    # LM = get_lines(args.LM)[i]
    # sim = get_lines('sim-score.txt')[i]
    input_sent = get_lines(args.sent)[i]
    input_context_sent = get_lines(args.context_sent)[i]


    sent  = scorer.sentence_score(input_sent, reduce="mean")
    context_sent  = scorer.sentence_score(input_context_sent, reduce="mean")


    sent_emb = model.encode(input_sent, convert_to_tensor=True)
    context_sent_emb = model.encode(input_context_sent, convert_to_tensor=True)

    # def cos_sim(a, b):
    #    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

    sim = cosine_scores = util.pytorch_cos_sim(sent_emb, context_sent_emb)
    sim = sim.cpu().numpy()
    sim = str(sim)[1:-1]
    sim = str(sim)[1:-1]
    print('sim', sim)
    score = Visual_re_ranker(sent, context_sent, sim)
    score = score.belief_revision()
    print('belief_revision', score)
    temp.append(score)

    result = str(score)
   

    f.write(result)
    f.write('\n')

f.close()
