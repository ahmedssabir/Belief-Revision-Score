## To get BR score for Karpathy split

First run ```LM-GPT-2.py ```   to ``LM.txt`` 

Then run this with the following flags: 

1. ``LM.txt``  to init the hyp

2. `` visual-context_label.txt`` class/object from the classifier 

3. ``visual-context.txt`` mean prob of all visual 

4. ``caption.txt`` from baseline 


```
python model.py --lm LM.txt  --vis visual-context_label.txt --vis_prob visual-context.txt --c caption.txt
```

