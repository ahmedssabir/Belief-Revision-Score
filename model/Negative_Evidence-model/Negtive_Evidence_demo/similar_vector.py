import gensim #4.0.1
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

model_path = 'glove.6B.300d.txt'

glove_file = datapath(model_path)

tmp_file = get_tmpfile("test_word2vec.txt")

_ = glove2word2vec(glove_file, tmp_file)

model = KeyedVectors.load_word2vec_format(tmp_file)

file1 = []



#with open('caption.txt','rU') as f1:
with open('visual_context.txt','rU') as f1:  



    for line1 in f1:

       file1.append(line1.rstrip())
       #break

resutl=[]



f=open('visual_context_negtive.txt', "w")
for i in range(len(file1)):
    temp =[]
    messages  = file1[i]
    print(messages)

    w = model.similar_by_vector(messages)[6]
    print(w)
     
    w = w[0]
    temp.append(w)

    result= file1[i]+','+str(w)

    f.write(result)
    f.write('\n')
    print(result)

f.close()





