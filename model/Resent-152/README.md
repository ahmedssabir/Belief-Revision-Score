## Extract visual information 
```
conda create -n Resnet python=3.7 anaconda
conda activate Resnet
pip install tensorflow==1.15.0
pip install keras==2.1.5
``` 

For [ResNet](https://arxiv.org/abs/1512.03385)

``` 
python run-visual.py
```

``` 
COCO_val2014_000000185210.jpg 'traffic_light', 0.7458004
COCO_val2014_000000235692.jpg  'ox', 0.49095494
``` 

For [CLIP](https://github.com/openai/CLIP) with zero-shot prediction

```
# torch 1.7.1 
conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.1
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git
```

run  

```
 python run-visual_CLIP.py
```

```
COCO_val2014_000000185210.jpg 'barrow', 0.0954
COCO_val2014_000000235692.jpg  'ox', 0.5092
```
