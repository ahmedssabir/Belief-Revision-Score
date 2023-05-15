## Semantric Relatendes with BERT  
Fine-tune BERT on the created  dataset. Please refer to [BERT-CNN](https://github.com/ahmedssabir/Textual-Visual-Semantic-Dataset)

### Requirements
- Tensorflow 1.15.0
- Python 2.7

```
conda create -n BERT_visual python=2.7 anaconda
conda activate BERT_visual
pip install tensorflow==1.15.0
``` 

```
python train_model_VC.py # train/val/and inference 
```
main page example
``` 
## relatedness score   

image: COCO_val2014_000000156242.jpg - Karpathy test split
```
```
BERT Base

('visual :', 'apple') # Visual (ours)
('caption :', 'a display of apple and orange at market')
('Prediction :', 0.9933211)
******
('visual :', 'apple') # Greedy 
('caption :', 'a fruit market with apples and orange')
('Prediction :', 0.98885113)
******
('visual :', 'apple') Beam Serach
('caption :', 'a fruit stand with apples and oranges')
('Prediction :', 0.9911321)

 BERT Large
 
('visual  :', 'apple')
('caption :', 'a display of apple and orange at market')
('Prediction :', 0.99782264)
****** 
('visual :', 'apple')
(''caption :', 'a fruit market with apples and orange')
('Prediction :', 0.99774504)
****** 
('visual :', 'apple')
('caption :', 'a fruit stand with apples and oranges')
('Prediction :', 0.9977704)
```
