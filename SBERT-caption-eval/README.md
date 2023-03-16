
## Standard Diversity 

**Div-n** and **mblue**  from [Speaking the Same Language: Matching Machine to Human Captions by Adversarial Training](https://github.com/rakshithShetty/captionGAN)

``` 
 python2 computeDivStats.py beam_sample.json
```

Vocabulary size

```
python voc_size.py --hyp beam.txt 
``` 
## Lexical Diversity 

First, [install the packages](https://pypi.org/project/lexicalrichness/) by``pip install lexicalrichness`` 

For Uniq tokens per caption  

```
Lexical Diversity/uniq_token.sh
``` 

For MTLD  

```
Lexical Diversity/MTLD.sh
``` 
For TTR  

```
Lexical Diversity/ttr.sh
``` 


other [diversity metrics](https://github.com/qingzwang/DiversityMetrics)





## Semantic Diversity 

``` 
python SBERT_eval_demo.py
``` 
output **0.7593903**

baseline 


| human ref | output caption |cosine score |
| ----------| ---------------|-------------| 
| closeup of two red-haired bulls with long horns. |  two bulls with horns standing next to each other |**0.7593903** |
| longhorn cattle with brown skin standing in a row. | two bulls with horns standing next to each other | 0.47357738 |
| an orange longhorn bull stands among the herd. | two bulls with horns standing next to each other |0.5293588 |
| a group of steers are standing and looking around.| two bulls with horns standing next to each other |0.308849 |
| long horn steer in standing in a cattle lot.| two bulls with horns standing next to each other | 0.5109421 |
|                                             |  **two bulls with horns standing next to each other** | **0.7593903**|  


our model 

| human ref | output caption |cosine score |
| ----------| ---------------|-------------| 
| closeup of two red-haired bulls with long horns. | two long horn bulls standing next to each other | **0.8167392** |
| longhorn cattle with brown skin standing in a row. | two long horn bulls standing next to each other|  0.4450949|
| an orange longhorn bull stands among the herd. | two long horn bulls standing next to each other | 0.4843703 |
| a group of steers are standing and looking around. |two long horn bulls standing next to each other |0.34269458 |
| long horn steer in standing in a cattle lot.| two long horn bulls standing next to each other |0.60156345 |
|                                             | **two long horn bulls standing next to each other** | **0.8167392** |
 
We take the **Max** score against all 5 human annotations.. then we AVERAGE(all_score_baseline) vs AVERAGE(all_score_visual_re-ranker)


To run your test set on standard karpathy testset

```
run_SDiv.sh
```
