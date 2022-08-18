#!/usr/bin/env python3
from doctest import OutputChecker
import sys
import argparse
import torch
import re
import os
import gradio as gr
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from lm_scorer.models.auto import AutoLMScorer as LMScorer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer('stsb-distilbert-base', device=device)
batch_size = 1
scorer = LMScorer.from_pretrained('gpt2' , device=device, batch_size=batch_size)


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


  
def Visual_re_ranker(caption, visual_context_label, visual_context_prob):
    caption = caption 
    visual_context_label= visual_context_label
    visual_context_prob = visual_context_prob
    caption_emb = model.encode(caption, convert_to_tensor=True)
    visual_context_label_emb = model.encode(visual_context_label, convert_to_tensor=True)


    sim =  cosine_scores = util.pytorch_cos_sim(caption_emb, visual_context_label_emb)
    sim = sim.cpu().numpy()
    sim = str(sim)[1:-1]
    sim = str(sim)[1:-1] 

    LM  = scorer.sentence_score(caption, reduce="mean")
    score = pow(float(LM),pow((1-float(sim))/(1+ float(sim)),1-float(visual_context_prob)))

    return LM, sim, score 



demo = gr.Interface(
    fn=Visual_re_ranker,
    description="Demo for Belief Revision based Caption Re-ranker with Visual Semantic Information",
    inputs=[gr.Textbox(value="a city street filled with traffic at night") , gr.Textbox(value="traffic"),  gr.Textbox(value="0.7458009")],
    outputs=[gr.Textbox(value="Language Model Score") , gr.Textbox(value="Semantic Similarity Score"),  gr.Textbox(value="Belief revision score via visual context")],

)
demo.launch()
