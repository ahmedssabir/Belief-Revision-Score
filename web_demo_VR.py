#!/usr/bin/env python3
from doctest import OutputChecker
import sys
import torch
import re
import os
import gradio as gr
import requests

#url = "https://github.com/simonepri/lm-scorer/tree/master/lm_scorer/models"
#resp = requests.get(url)

from sentence_transformers import SentenceTransformer, util
#from sentence_transformers import SentenceTransformer, util
#from sklearn.metrics.pairwise import cosine_similarity
#from lm_scorer.models.auto import AutoLMScorer as LMScorer
#from sentence_transformers import SentenceTransformer, util
#from sklearn.metrics.pairwise import cosine_similarity

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
#model_sts = gr.Interface.load('huggingface/sentence-transformers/stsb-distilbert-base') 

#model_sts = SentenceTransformer('stsb-distilbert-base')
model_sts = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
#batch_size = 1
#scorer = LMScorer.from_pretrained('gpt2' , device=device, batch_size=batch_size)

#import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import re

def Sort_Tuple(tup):  
  
	# (Sorts in descending order)  
	tup.sort(key = lambda x: x[1])  
	return tup[::-1]


def softmax(x):
	exps = np.exp(x)
	return np.divide(exps, np.sum(exps))
	

def get_sim(x):
    x =  str(x)[1:-1]
    x =  str(x)[1:-1]
    return x
     
# Load pre-trained model 

model = GPT2LMHeadModel.from_pretrained('gpt2', output_hidden_states = True, output_attentions = True)

#model  =  gr.Interface.load('huggingface/distilgpt2', output_hidden_states = True, output_attentions = True)

#model.eval()
#tokenizer =  gr.Interface.load('huggingface/distilgpt2')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')


def cloze_prob(text):

	whole_text_encoding = tokenizer.encode(text)
	# Parse out the stem of the whole sentence (i.e., the part leading up to but not including the critical word)
	text_list = text.split()
	stem = ' '.join(text_list[:-1])
	stem_encoding = tokenizer.encode(stem)
	# cw_encoding is just the difference between whole_text_encoding and stem_encoding
	# note: this might not correspond exactly to the word itself
	cw_encoding = whole_text_encoding[len(stem_encoding):]
	# Run the entire sentence through the model. Then go "back in time" to look at what the model predicted for each token, starting at the stem.
	# Put the whole text encoding into a tensor, and get the model's comprehensive output
	tokens_tensor = torch.tensor([whole_text_encoding])
	
	with torch.no_grad():
		outputs = model(tokens_tensor)
		predictions = outputs[0]   

	logprobs = []
	# start at the stem and get downstream probabilities incrementally from the model(see above)
	start = -1-len(cw_encoding)
	for j in range(start,-1,1):
			raw_output = []
			for i in predictions[-1][j]:
					raw_output.append(i.item())
	
			logprobs.append(np.log(softmax(raw_output)))
			
	# if the critical word is three tokens long, the raw_probabilities should look something like this:
	# [ [0.412, 0.001, ... ] ,[0.213, 0.004, ...], [0.002,0.001, 0.93 ...]]
	# Then for the i'th token we want to find its associated probability
	# this is just: raw_probabilities[i][token_index]
	conditional_probs = []
	for cw,prob in zip(cw_encoding,logprobs):
			conditional_probs.append(prob[cw])
	# now that you have all the relevant probabilities, return their product.
	# This is the probability of the critical word given the context before it.

	return np.exp(np.sum(conditional_probs))





def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


  
def Visual_re_ranker(caption_G, caption_B, caption_VR, visual_context_label, visual_context_prob):
    caption_G = caption_G
    caption_B = caption_B
    caption_VR = caption_VR
    visual_context_label= visual_context_label
    visual_context_prob = visual_context_prob
    caption_emb_G = model_sts.encode(caption_G, convert_to_tensor=True)
    caption_emb_B = model_sts.encode(caption_B, convert_to_tensor=True)
    caption_emb_VR = model_sts.encode(caption_VR, convert_to_tensor=True)

    visual_context_label_emb = model_sts.encode(visual_context_label, convert_to_tensor=True)


    sim_1 =  cosine_scores = util.pytorch_cos_sim(caption_emb_G, visual_context_label_emb)
    sim_1 = sim_1.cpu().numpy()
    sim_1 = get_sim(sim_1)

    sim_2 = cosine_scores = util.pytorch_cos_sim(caption_emb_B, visual_context_label_emb)
    sim_2 = sim_2.cpu().numpy()
    sim_2 = get_sim(sim_2)

    sim_3 = cosine_scores = util.pytorch_cos_sim(caption_emb_VR, visual_context_label_emb)
    sim_3 = sim_3.cpu().numpy()
    sim_3 = get_sim(sim_3)
 

    LM_1 = cloze_prob(caption_G)
    LM_2 = cloze_prob(caption_B)
    LM_3 = cloze_prob(caption_VR)

    #LM  = scorer.sentence_score(caption, reduce="mean")
    score_1 = pow(float(LM_1),pow((1-float(sim_1))/(1+ float(sim_1)),1-float(visual_context_prob)))
    score_2 = pow(float(LM_2),pow((1-float(sim_2))/(1+ float(sim_2)),1-float(visual_context_prob)))
    score_3 = pow(float(LM_3),pow((1-float(sim_3))/(1+ float(sim_3)),1-float(visual_context_prob)))

    #return {"LM": float(LM)/1, "sim": float(sim)/1, "score": float(score)/1 }
    return {"Graddy": float(score_1)/1, "Best-Beam-5": float(score_2)/1, "Visual_re-Ranker": float(score_3)/1  }
    #return LM, sim, score 




demo = gr.Interface(
    fn=Visual_re_ranker,
    description="Demo for Belief Revision based Caption Re-ranker with Visual Semantic Information",
    #inputs=[gr.Textbox(value="a city street filled with traffic at night") , gr.Textbox(value="traffic"),  gr.Textbox(value="0.7458009")],
    inputs=[gr.Textbox(value="a longhorn cow with horns standing in a field") , gr.Textbox(value="two bulls with horns standing next to each other"), gr.Textbox(value="two bulls standing next to each other"), gr.Textbox(value="ox"),  gr.Textbox(value="0.49095494")],
    #outputs=[gr.Textbox(value="Language Model Score") , gr.Textbox(value="Semantic Similarity Score"),  gr.Textbox(value="Belief revision score via visual context")],
    outputs="label",
)
demo.launch()