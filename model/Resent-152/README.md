
## Extract visual information 



For [SwinV2](https://arxiv.org/abs/2111.09883)

``` 
conda create -n SwinV2 python=3.10 anaconda
conda activate SwinV2
pip install transformers==4.29.0 
``` 

```
python run_swinv2_visual.py  --input image file --output output results
```

```
COCO_val2014_000000185210.jpg  ('visual', 'traffic light, traffic signal, stoplight') Prob:0.6626558303833008
COCO_val2014_000000235692.jpg  ('visual', 'ox') Prob:0.867093026638031
```

For [ViT](https://arxiv.org/abs/2010.11929)

``` 
conda create -n vit python=3.10 anaconda
conda activate vit
pip install transformers==4.29.0 
``` 


```
python run_ViT_visual.py --input image file --output output results
```

```
COCO_val2014_000000185210.jpg  ('visual:', 'traffic light, traffic signal, stoplight') Prob:0.8653415441513062
COCO_val2014_000000235692.jpg  ('visual:', 'ox') Prob:0.8389703035354614
```




For [ResNet](https://arxiv.org/abs/1512.03385)

```
conda create -n Resnet python=3.7 anaconda
conda activate Resnet
pip install tensorflow==1.15.0
pip install keras==2.1.5
``` 

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

```
 python run-visual_CLIP.py
```

```
COCO_val2014_000000185210.jpg 'streetcar', 0.2280
COCO_val2014_000000235692.jpg  'ox', 0.5092
```



