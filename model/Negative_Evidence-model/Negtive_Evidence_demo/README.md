## Visual Context with Negtive Evidence

We extract top-20 beam search from SOTA [Caption Generator](https://github.com/aimagelab/meshed-memory-transformer)  [1]``caption.txt``


<img align="center" width="400" height="200" src="COCO_val2014_000000235692.jpg">


````
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
```` 


To use this approach, first, we need to extract the negative information from the visual context. We will use GloVe to extract objects with high confidence ``from the visual classifier`` . Then we use the detected objects as query to a pre-trained 840B GloVe and retrieve the close concepts, and then, we employ the objects are not detected/present in the image as negative evidence.


To extract visual information we run the visual classifier  
```
python /model/Resnt152/run-visual.py 
``` 
Then the output _label class_ ``ox`` is used as **query** to retrieve close concepts. 

```  
python similar_vector.py
``` 

The negative visual information from GloVe is ``goat`` and  it will be used to decrease the hypothesis. Next, we need to init that negative visual context with common observation (LM) 

``` 
python negative_visual_context_inti_GPT2.py
```

and finally 
``` 
 python Example_negtive_2/negative_evidence_SBERT.py --lm LM.txt --visNg negtive_visual.txt  --visNg_init negtive_visual_init.txt --c caption.txt
```

Remove beam search duplicate caption or see re-ranked output file ``Belief-revision-Neg_Ev_re-rank.txt`` 
```
sort -u -o Belief-revision-with-Negative_Evidence.txt  Belief-revision-with-Negative_Evidence.txt
```
Re-ranking with the highest score (caption --> related visual context)
```
sort -t, -rk3 Belief-revision-with-Negative_Evidence.txt
```


Visual re-ranked beams serach  B = 5 with **best** visual beam = 3 

``` 
a couple of bulls standing next to each other 0.27912633722739355
two bulls standing next to each other 0.26619051767139323
two long horn bulls standing next to each other 0.22234478804409874
two bulls with horns standing next to each other 0.20839996352634216
a longhorn cow with horns standing in a field 0.033621208853031326
```

#### References
[1] Cornia, Marcella, et al. "Meshed-memory transformer for image captioning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
