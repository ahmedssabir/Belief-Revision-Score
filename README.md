

# Belief Revision based Caption Re-ranker with Visual Semantic Information



<img align="right" width="600" height="200" src="overview.png">
In this work, we propose a re-ranking approach to improve the performance of caption generation systems. We leverage on visual semantic measures to match the proper caption to its related visual information in the image (e.g object information). The re-ranker uses human inspired Belief Revision (Blok et. al. 2003) to revise the caption original likelihood using the semantic relatedness via visual context from the image.  Experiments show that adding visual information can improve the performance of the captioning system without re-training or fine-tuning. Additionally, we investigate whether a simple semantic similarity-based metric SBERT-sts for caption evaluation can capture similarities better than word n-gram based metrics. 
<br/>
<br/>

This repository contains the  implementation of the paper [Re-ranking Caption Generation with Visual Context Semantics](https://arxiv.org)




##  Contents
0. [Overview](#overview)
1. [Visual Re-ranking with Belief Revision](#Visual-Re-ranking-with-Belief-Revision)
2. [Dataset](#dataset)
3. [Model](#Model)
4. [Visual Re-ranking with Negative Evidence](#Visual-Re-ranking-with-Negative-Evidence)
5. [Semantic Diversity Evaluation](#Semantic-Diversity-Evaluation)
6. [Cloze Prob based Belife Revision](#Cloze-Prob-based-Belife-Revision)
7. [Other Task: Sentence Semantic Similarity](#Other-Task-Sentence-Semantic-Similarity)
8. [Citation](#Citation)



## Visual Re-ranking with Belief Revision 
The [Belief revision](https://www.aaai.org/Papers/Symposia/Spring/2003/SS-03-05/SS03-05-005.pdf) is  a conditional probability model which assumes that the preliminary probability finding is revised to the extent warranted by the hypothesis proof.  

<img src="https://render.githubusercontent.com/render/math?math=\text{P}(w \mid c)=\text{P}(w)^{\alpha}"> 


where the main components of hypothesis revision as caption visual semantics re-ranker:

1. Hypothesis (caption candidates beam search) <img src="https://render.githubusercontent.com/render/math?math=\text{P}(w)"> initialized by common observation (ie language model) 
   
2. Informativeness  <img src="https://render.githubusercontent.com/render/math?math=1-\text{P}(c)"> of the visual context from the image
 
3. Similarities <img src="https://render.githubusercontent.com/render/math?math=\alpha=\left[\frac{1 - \text{sim}(w, c)}{1%2B\text{sim}(w, c)}\right]^{1-\text{P}(c)}"> the relatedness between the two concepts (visual context and hypothesis) with respect to the informativeness of the visual information.
 
Here is a [Demo](visual_re-re-ranker_demo.ipynb) to show the Visual Re-ranking based Belief Revision 
  
## Dataset
We enrich COCO-caption with **textual Visual Context** information. We use [ResNet152](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf) to extract
 object information for each COCO-caption image.
 
 VC1     | VC2   |  VC3   | human annoated caption     | 
| ------------- | ------------- |------------- | ------------- |
| cheeseburger  | plate       |  hotdog   |    a plate with a hamburger fries and tomatoes   |
| bakery  | dining table       |  website   |    a table having tea and a cake on it   |
| gown  | groom       |  apron   |    its time to cut the cake at this couples wedding   |


More information about the visual context extraction  [paper](https://github.com/ahmedssabir/dataset)



## Model 
Here, we describe in more detail the impalement of belief revision as a visual re-ranker. We show that by
integrating visual context information, a more descriptive caption is re-ranked higher. 
Our model can  be used as a drop-in complement for any caption generation algorithm 
that outputs a list of candidate captions

To run the **B**elief **R**evision-**S**core via visual context directly with GPT-2 ans SRoBERTa-sts

```
conda create --name BRscore  python=3.7
source activate BRscore
# tested with sentence_transformers-2.2.0
pip install sentence_transformers 
```
run with GPT-2+SRoBERTa 

```
python model.py  --c model/Example_1/caption.txt --vis model/Example_1/visual_context_label.txt --vis_prob model/Example_1/visual_context_prob.txt 
```
Also interactive demo with huggingface-gardio by running this code or [colab](https://colab.research.google.com/drive/1JGPvyHptI65SDXXZNp75YdU_T3GdLVRs?usp=sharing) here

```
pip install gradio 
python demo.py 
```




To run each step separately, which gives you the flexibility to use different SoTA model (or your custom model)

First, we need to initialize the hypothesis with common observation ie lanaguage model [(GPT2)](https://github.com/simonepri/lm-scorer)


```
conda create -n LM-GPT python=3.7 anaconda
conda activate LM-GPT
pip install lm-scorer
python model/LM-GPT-2.py 
``` 
Second, we need the visual context from the image 
````
python  model/run-visual.py
```` 

Finally, the relatendes between the two conecpt (visual context and hypothesis) 

Using fine-tuning BERT
 [fine-tuning BERT](BERT/)
```
python BERT/train_model_VC.py 
```
Or [general-purpose SBERT](https://github.com/UKPLab/sentence-transformers) with cosine similairy 

```
conda create -n SBERT python=3.7 anaconda
conda activate SBERT
pip install sentence-transformers
python model/SBERT_model_VC.py
```
Then run **demo** Example 1/2 (below)

```
 python model/Example_1/model.py --lm LM.txt --vis visual_context_lable.txt --vis_prob visual_context_prob.txt --c caption.txt
``` 
Note that each score is computed separately here (each score is in a separate file)

Go here for [more details](https://github.com/sabirdvd/BERT-visual-caption-/tree/main/model) 

## Demo 
Here an examples with SBERT based model 

### Example 1 

<img align="center" width="400" height="200" src="example.jpg">


Baseline beam = 5
```
a city street filled with traffic at night       	 
a city street covered in snow at night	 
a city street covered in traffic at night time	 
a city street filled with traffic surrounded by tall buildings	 
a city street covered in traffic at night	 
```





Visual re-ranked beam serach  = 5 

``` 
a city street filled with traffic surrounded by tall buildings 
a city street filled with traffic and traffic lights 
a city street filled with traffic surrounded by snow 
a city street filled with traffic at night 
a city street at night with cars and street lights 
```

We re-ranked the best 5 beams from 9 candidates captions, generated by the baseline, using the visual context information.


### Example 2


<img align="center" width="400" height="200" src="example-2.jpg">


Baseline beam = 5
```
a longhorn cow with horns standing in a field 
two bulls standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other
```

Visual re-ranked beam serach  = 5   
``` 
two bulls standing next to each other 
a couple of bulls standing next to each other 
two bulls with horns standing next to each other 
two long horn bulls standing next to each other 
a longhorn cow with horns standing in a field 
```
We re-ranked the best 5 beams from 20 candidates captions, generated by the baseline, using the visual context information.


## Visual Re-ranking with Negative Evidence 

Until now, following the same concept we considered only the cases when the visual context increase
the belief of the hypothesis. However the same [work](https://link.springer.com/content/pdf/10.3758/BF03193607.pdf) proposed another idea  for the case where the absence of evidence leads to decrease the hypothesis probability.


<img src="https://render.githubusercontent.com/render/math?math=\text{P}(w \mid \neg c)=1-(1-\mathrm{P}(w))^{\alpha}">

In our case we tries to introduce negative evidence in two ways: **(1)** objects detected by Resnet
with a very low confidence are assumed not to be actually in the image and used as negative evidence,
and **(2)** using the objects actually detected with
high confidence to query pre-trained 840B [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)  and retrieve close concepts, which are used as negative evidence (after checking they are not detected in the image).

## Example 1 with Negative Evidence

<img align="center" width="400" height="200" src="example.jpg">


In this example, we will use the **second method** with Pre-trained [Glove](https://nlp.stanford.edu/projects/glove/) Vector to extract the negative information related to the visual context but not detected in the image.

``` 
conda create -n GloVe python=3.8 anaconda
conda activate GloVe
pip install gensim==4.1.0
python Negative_Evidence-model/similar_vector.py
``` 

The  negative visual information is  ``accident``  that will be used to decrease the hypothesis. Note that the negative visual information also needs to be initialized by common observations  `` python model/LM-GPT-2.py `` 


```
python Example_negtive_1/negative_evidence_SBERT.py --lm LM.txt --visNg negtive_visual.txt  --visNg_init negtive_visual_init.txt --c caption.txt
```


Baseline beam = 5
```
a city street filled with traffic at night       	 
a city street covered in snow at night	 
a city street covered in traffic at night time	 
a city street filled with traffic surrounded by tall buildings	 
a city street covered in traffic at night	 
```

Visual re-ranked beam serach  = 5 with negative evidence   
```
a city street filled with traffic surrounded by tall buildings
a city street filled with traffic surrounded by snow
a city street filled with traffic and traffic lights
a city street filled with traffic at night
a city street at night with cars and street lights
``` 

We re-ranked the best 5 beams from 9 candidates captions, generated by the baseline, using a negtive visual context information.

For more [example](https://github.com/sabirdvd/BERT-visual-caption-/tree/main/model/Negative_Evidence-model/Negtive_Evidence_demo)


 ## Semantic-Diversity-Evaluation
 
 ### Sentence-to-sentence semantic similarity for semantic diversity evaluation
Inspired by  [BERTscore ](https://github.com/Tiiiger/bert_score), we propose sentence-to-sentence semantic similarity score to
compare candidate captions with human references. We employ pre-trained [Sentence-RoBERTa-L ](https://www.sbert.net/docs/pretrained_models.html) tuned for general STS task. SBERT-sts uses a siamese network to
derive meaningful sentence embeddings that can
be compared via cosine similarity.

For more deatil and [other diversity evaluation](SBERT-caption-eval)
 
 Example 
                                                                                    
 Model     | caption   |  BERTscore   | SBERT-sts*    |  Human subject |
| ------------- | ------------- |------------- | ------------- | ------------- |
| B-best  | two bulls with horns standing next to each other      |  0.89   |    0.75    |   16.7 |   
| B+VR    | two long horn bulls standing next to each other  |0.88   |    <b>0.81</b>    | <b>0.83</b> |
| Human      | a closeup of two red-haired bulls with long horns       |  | 

(*) _max(sim(ref_k, candidate caption))_, k = 5 human references 

To find the cosine score 
```
python SBERT-caption-eval/SBERT_eval_demo.py --ref ref_demo.txt --hyp hyp-demo_BeamS.txt or hyp-demo_visual_re-ranked.txt
```

## Cloze Probability based Belife Revision 
Cloze probability is the probability that a given word will be produced in a given context on
a sentence completion task (last word).

```
The girl eats the **toast**  --> low probability 
The girl eats the **eggs** --> high probability 
```

```
cloze_prob('The girl eats the toast')
0.0004592319609081579

cloze_prob('The girl eats the eggs')
0.00436875504749275
``` 
For caption re-ranking 

```
cloze_prob('a city street at night with cars and street lamps')
0.12100567314071672

cloze_prob('a city street filled with traffic and traffic lights')
0.40925383021394385
```

Cloze_Prob based Belife revision

```
a city street at night with cars and street lamps 0.18269163340435274
a city street filled with traffic and traffic lights 3.0824517390664777e-16
```
The first sentence is more diverse and without any repetition of word traffic.

## Other Task: Sentence Semantic Similarity

```
conda create --name BR-S  python=3.7
# test with sentence_transformers-2.2.0
pip install sentence_transformers 
```
run 

```
python  sent_model/model_sent.py --sent sent.txt --context_sent context_sent.txt  --output score.txt
```
Exp:
```
sent =  'Obama speaks to the media in Illinois' 
context_sentence =  'The president greets the press in Chicago'
```
The two sentences is related but not similar and belief revision score capture the relatedness better than semantic similarity.


```
# SBERT cosine 
Cosine = 0.62272817

# belief_revision score 
belief_revision = 0.557584688720967
``` 
1) belief_revision_score balances the high similarity score using human-inspired logic understanding.  The similarity cosine distance alone is not a reliable score in some scenarios as it measures the angle between vectors in the semantic space.
 

2) The output is a probability that can be used to re-rank or combined with another score. Note that with a cosine distance you cant do that.


For quick start a [colab demo](https://colab.research.google.com/drive/1ipTLmZxLLU5aNUQQvSHJRrsetQpg_31C?usp=sharing)

## Citation

The details of this repo are described in the following paper. If you find this repo useful, please kindly cite it:

```bibtex
@inproceedings{XXXXXX,
  title={ZZZZZZ},
  author={XXXXX},
  booktitle={XXXX},
  year={2021}
}
```

### Acknowledgement
The implementation of the Belief Revision Score relies on resources from <a href="https://github.com/simonepri/lm-scorer">lm-score</a>, <a href="https://github.com/huggingface/transformers">Huggingface Transformers</a>, and <a href="https://www.sbert.net/">SBERT</a>. We thank the original authors for their well organized codebase.
