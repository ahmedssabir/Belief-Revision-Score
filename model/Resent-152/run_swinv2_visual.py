# -*- coding: utf-8 -*-
import torch
from transformers import ViTImageProcessor
from PIL import Image
import os
import glob
import argparse
from transformers import AutoImageProcessor, Swinv2ForImageClassification
import torch



parser=argparse.ArgumentParser(description='extract visual context')
parser.add_argument('--input',  default='visual_context.txt', help='visual_context', type=str,required=True)   
parser.add_argument('--output',  default='visual_context.txt', help='visual_context', type=str,required=True)   
args = parser.parse_args()


#image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
#model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")#

#image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

#image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
#model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

#filenames = glob.glob("all/*.jpg")
filenames = glob.glob(args.input)
filenames.sort()
for image in filenames:

  input_image = Image.open(image)
 
  image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
  model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
  inputs = image_processor(input_image, return_tensors="pt")
  
    
  with torch.no_grad():
     logits = model(**inputs).logits

  predicted_label = logits.argmax(-1).item()
  print(model.config.id2label[predicted_label])
  output = "Predicted class:", model.config.id2label[predicted_label]
    
   
  prob = torch.softmax(logits, dim=1)
  top_p, top_class = prob.topk(1, dim = 1)
  print (top_p)
  

  print('visual: '+str(output) + ' ' + str(top_p))

 
  with open(args.output, 'a') as fp:
      #fp.write(str(output) + ' ' + str(top_p.item()))
      fp.write(str(output) + ' ' + "Prob:" + str(top_p.item()))
      fp.write('\n')
