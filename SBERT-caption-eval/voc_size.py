
#!/usr/bin/env python3
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--hyp', default='', help='output', type=str,required=False)  
args = parser.parse_args()



with open(args.hyp) as f:
    texts = f.read()
    #print(contents)

def count_vocab(text):

    # Normalize the text and get the vocabulary size
    tokens = list(set(text.lower().split()))

    # Remove all tokens that are not alphabetic
    words = [word for word in tokens if word.isalpha()]

    vocab_size = len(words)

    return vocab_size


print(count_vocab(texts))

