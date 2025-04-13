# -*- coding: utf-8 -*-
import torch
from transformers import AutoImageProcessor, Swinv2ForImageClassification
from PIL import Image
import glob
import argparse
import os


parser = argparse.ArgumentParser(description='Extract visual context')
parser.add_argument('--input', default='images/*.jpg', help='Input image pattern', type=str, required=True)
parser.add_argument('--output', default='visual_context.txt', help='Output result file', type=str, required=True)
args = parser.parse_args()


image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

#image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
#model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")#

#image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")
#model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256")

#image_processor = AutoImageProcessor.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")
#model = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-large-patch4-window12-192-22k")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()


filenames = glob.glob(args.input)
filenames.sort()


total_images = len(filenames)
print(f"Total images: {total_images}")


with open(args.output, 'w') as fp:
    fp.write("Image\tObject\tProbability\n")


for image_path in filenames:
    input_image = Image.open(image_path).convert("RGB")
    inputs = image_processor(input_image, return_tensors="pt").to(device)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_label_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_label_id]
    prob = torch.softmax(logits, dim=1)
    top_prob = prob[0][predicted_label_id].item()

    print(f"Image: {os.path.basename(image_path)}, Object: {predicted_label}, Prob: {top_prob:.4f}")

    
    with open(args.output, 'a') as fp:
        fp.write(f"{os.path.basename(image_path)}\t{predicted_label}\t{top_prob:.4f}\n")


print(f"\nProcessed {total_images} images. Results saved to '{args.output}'.")
