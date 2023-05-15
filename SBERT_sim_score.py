import argparse
import re
import numpy 
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

#2021 model 
# https://www.sbert.net/docs/pretrained_models.html
#model = SentenceTransformer('paraphrase-mpnet-base-v2')

# how to run
#python SBERT_sim_score.py  --caption caption.txt --vis visual_context_label.txt


parser=argparse.ArgumentParser()
parser.add_argument('--caption',  default='caption.txt', help='beam serach', type=str,required=True)  
parser.add_argument('--vis', default='visual_context_label.txt', help='visual_context from ResNet', type=str,required=True)  
args = parser.parse_args()

#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')


def cos_sim(a, b):
    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))


def get_lines(file_path):
    with open(file_path) as f:
            return f.read().strip().split('\n')

 
output_path = 'caption-visual_context_score.txt'

# compute visual context
f=open(output_path, "w")
for i in range(len(get_lines(args.caption))):
    temp =[]
    caption  = get_lines(args.caption)[i]
    visual_context_label = get_lines(args.vis)[i]
   
    
    #Compute embedding for both lists
    caption_emb = model.encode(caption, convert_to_tensor=True)
    visual_context_label_emb = model.encode(visual_context_label, convert_to_tensor=True)

    #def cos_sim(a, b):
    #    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))

   
    sim =  cosine_scores = util.pytorch_cos_sim(caption_emb, visual_context_label_emb)
    #print(w)
  
    
    sim = sim.cpu().numpy()
    sim = sim.item()
    temp.append(sim)

    # to print visual-context,caption,score
    #result= file1[i]+','+file2[i]+','+str(w)
    
    # score only 
    result= str(sim)
    
    f.write(result)
    f.write('\n')
    
    
f.close()
