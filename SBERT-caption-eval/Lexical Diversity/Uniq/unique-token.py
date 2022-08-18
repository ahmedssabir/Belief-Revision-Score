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
 
    lex = LexicalRichness(messages)

    w = lex.terms
    
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





