
import os
import glob
import sys
import torch
import torchvision.transforms as Transforms
import clip
from PIL import Image



# Check device
#device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device("cpu")
print(f"Device - {device}")

# Load CLIP model
clip_model, clip_preprocess = clip.load('ViT-B/32', device)
clip_model.eval()

#
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]

text = clip.tokenize(categories).to(device)

def predict_clip(image_file_path):
    image = clip_preprocess(Image.open(image_file_path)).unsqueeze(0).to(device)
    # base model for the bigger model ViT-L/14 
    clip_model, _ = clip.load('ViT-B/32', device)

    # Calculate features
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    predictions = {}
    for value, index in zip(values, indices):
        predictions[f"{categories[index]:>16s}"] = f"{1 * value.item():.4f}%"
	
    return predictions


# run pred 
filenames = glob.glob("file= '/image/*.jpg")
filenames.sort()
for image in filenames:
     print(os.path.basename(image), predict_clip(image))
#print(predict_clip("image.jpg"))



