
### Visual semantic Re-ranker with example 

In this example we extract top-20 beam search from SOTA [Caption Generator](https://github.com/aimagelab/meshed-memory-transformer) ``caption.txt``

<img align="center" width="400" height="200" src="COCO_val2014_000000235692.jpg">




```
a longhorn cow with horns standing in a field
two bulls standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
two bulls with horns standing next to each other	 
a couple of bulls standing next to each other	 
a couple of bulls standing next to each other	 
two long horn bulls standing next to each other	 
two long horn bulls standing next to each other	 
two long horn bulls standing next to each other	 
two long horn bulls standing next to each other
two long horn bulls standing next to each other	
two long horn bulls standing next to each other
two long horn bulls standing next to each other
two long horn bulls standing next to each other
```

To use this approach, first, we need to initialize  the hypothesis with common observation (GPT2) ``LM.txt``

```
python LM-GPT-2.py 
``` 

Secondy, we need the [confidence score](https://github.com/ahmedssabir/Belief-Revision-Score/tree/main/model/Resent-152) from the classifier.  `visual_context_Resent.txt`

``Another option``: to initialize the visual context by (1) common observation (GPT2) or (2) [Movie  OpenSubtitle corpus](http://www.lrec-conf.org/proceedings/lrec2016/pdf/947_Paper.pdf) unigram language model. 

```
python /model/Resent-152/run-visual.py
COCO_val2014_000000235692.jpg [('ox', 0.49095494)]
``` 
`` option 2``: To initialize both with GPT2 (hyp and the visual context) 
```
python LM-GPT-2.py && python VC-GPT-2.py
```  

Finally, the relatedness between the two concepts (visual context and hypothesis) using [fine-tuning BERT](https://github.com/ahmedssabir/Belief-Revision-Score/tree/main/BERT).

```
python BERT/train_model_VC.py 
```
Or general-purpose [Sentence BERT](https://github.com/UKPLab/sentence-transformers) RoBERTa-L with cosine similarity.

```
 python SBERT_sim_score.py --caption caption.txt  --vis visual_context_label.txt 
```

Or recent model in semantic similarity [Contrastive Learning of Sentence Embeddings](https://github.com/princeton-nlp/SimCSE). 

```
 python SimCSE_sim_score.py --caption caption.txt  --vis visual_context_label.txt 
``` 

Or SoTA model in semantic similarity [InfoCSE: Information-aggregated Contrastive Learning of Sentence Embeddings](https://github.com/caskcsg/sentemb/tree/main/InfoCSE)

```
 python model_Info_CSE.py --caption caption.txt  --vis visual_context_label.txt 
``` 

After having all the required files 


- `LM.txt`: initialized hypothesis by common observation (_i.e._ LM)
- `visual-context_label.txt`: visual context from the classifier 
- `visual_context_prob.txt`: initialized visual context or classifer confident (prob)
- `sim.txt`:   the similarity between the concept (BERT_fine_tune/SBERT,etc)
- `caption.txt`: beam search candidates caption from the baseline


to run this example with SBERT similarity 

``` 
python Example_2/model.py --lm LM.txt --vis visual_context_lable.txt --vis_prob visual_context_prob.txt --c caption.txt
```
or with SimCSE-BERT similarity 

```
python SimCSE-BERT/python model_SimCSE.py --lm LM.txt --vis visual_context_lable.txt --vis_prob visual_context_prob.txt --c caption.txt
``` 
or with Info_CSE-BERT similarity 

```
python Info_CSE-BERT/model_Info_CSE.py --lm LM.txt  --vis visual_context_lable.txt  --vis_prob visual_context_prob.txt --c caption.txt
```

Remove beam search duplicate caption or check the re-ranked output file `` Belief-revision_re-rank.txt`` 
```
sort -u -o Belief-revision.txt  Belief-revision.txt
```
Re-rank with the highest score (caption --> related visual context)
```
sort -t, -rk3 Belief-revision.txt
```

Visual re-ranked beams serach  B = 5 with **best** visual beam = 4 
```
two bulls standing next to each other 0.31941289259462063
a couple of bulls standing next to each other 0.2858426977047663
two bulls with horns standing next to each other 0.26350009525262974
two long horn bulls standing next to each other 0.24074783064577798
a longhorn cow with horns standing in a field 0.0.03975113398536263
 ``` 
In this example, we re-ranked the top 20 related captions to their visual context with visual semantics. 


