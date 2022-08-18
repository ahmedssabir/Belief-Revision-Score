#!/usr/bin/env python3
import sys
import argparse

from lexicalrichness import LexicalRichness 

parser=argparse.ArgumentParser()
parser.add_argument('--hyp', default='beam_search.txt', help='caption baseline', type=str,required=False)  
parser.add_argument('--output', default='uniq_word_pre_caption.txt', help='result', type=str,required=False)  
args = parser.parse_args()  

file1 = []

with open(args.hyp,'rU') as f:
    for line in f:
       file1.append(line.rstrip())



output_path= 'score.txt'
f=open(output_path, "w")

for i in range(len(file1)):
    temp =[]
    messages  = file1[i]
    #messages1 = file2[i]
    #messages  = "people read the book"
    #messages1 = "man in the street eating hotdog"
  
    #tf.logging.set_verbosity(tf.logging.ERROR)
    #flt = ld.flemmatize(messages)
    lex = LexicalRichness(messages)
    #w = lex.mtld(threshold=0.72)
    w = lex.ttr 
    #w = lex.terms
    #Compute embedding for both lists
    #embeddings1 = model.encode(messages, convert_to_tensor=True)
    #embeddings2 = model.encode(messages1, convert_to_tensor=True)

    #def cos_sim(a, b):
    #    return np.inner(a, b) / (np.linalg.norm(a) * (np.linalg.norm(b)))
    #w = de2en.translate(messages) 
    #w = mtld((messages.split()))
    #w = hdd((messages.split()))
    #w =  cosine_scores = util.pytorch_cos_sim(embeddings1, embeddings2)
    #tf.reset_default_graph()
  

    
    temp.append(w)

    #result= file1[i]+','+file2[i]+','+str(w)
    result= str(w)

    f.write(result)
    #f.write(result)
    f.write('\n')
    print(result)
    #del result
    #close.sess()
    
f.close()





