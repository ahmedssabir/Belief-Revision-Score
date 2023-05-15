# -*- coding: utf-8 -*-
from transformers import ViTForImageClassification
import torch
from transformers import ViTImageProcessor
from PIL import Image
import os
import glob
import argparse

# pip install transformers==4.29.0  

parser=argparse.ArgumentParser(description='extract visual context')
parser.add_argument('--input',  default='visual_context.txt', help='visual_context', type=str,required=True)   
parser.add_argument('--output',  default='visual_context.txt', help='visual_context', type=str,required=True)   
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224')
model.to(device)


#filenames = glob.glob("/Users/asabir/Dropbox/weekly-meeting/2023/abril-2023/Flicker-30K-data/extracted_images-1k/*.jpg")
filenames = glob.glob(args.input)
#filenames = glob.glob("COCO_val2014_000000185210.jpg")
#filenames = glob.glob("all/*.jpg")

filenames.sort()
for image in filenames:

  input_image = Image.open(image)
  processor = ViTImageProcessor.from_pretrained('google/vit-large-patch16-224')
  inputs = processor(images=input_image, return_tensors="pt").to(device)
  pixel_values = inputs.pixel_values


  with torch.no_grad():
    outputs = model(pixel_values)
    logits = outputs.logits
    logits.shape

  prediction = logits.argmax(-1)
  print("visual:", model.config.id2label[prediction.item()])
  output = "visual:", model.config.id2label[prediction.item()]
   
   
  prob = torch.softmax(logits, dim=1)
  top_p, top_class = prob.topk(1, dim = 1)
  #print (top_p)

  print('visual: '+str(output) + ' ' + str(top_p.item()))

 
  with open(args.output, 'a') as fp:
      fp.write(str(output) + ' ' + "Prob:" + str(top_p.item()))
      fp.write('\n')
      
       