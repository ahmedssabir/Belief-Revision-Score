
## Cloze Probability based Belife Revision 
Cloze probability is the probability of a given word that will be filled in a given context on a sentence completion task (last word).

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

To run 

```
python model_coze.py  --c caption.txt --vis visual_context_label.txt --vis_prob visual_context_prob.txt 
```



We also support Japanese language 

```
python model_coze_jp.py  --c caption_jp.txt --vis visual_context_label_jp.txt --vis_prob visual_context_prob.txt
```
