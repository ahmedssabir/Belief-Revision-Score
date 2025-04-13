import os
import glob
import argparse
import torch
import clip
from PIL import Image


parser = argparse.ArgumentParser(description="CLIP image classification")
parser.add_argument('--input', default='images/*.jpg', help='Input image pattern', type=str, required=True)
parser.add_argument('--output', default='clip_output.txt', help='Output result file', type=str, required=True)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device - {device}")


clip_model, clip_preprocess = clip.load('ViT-B/32', device)
clip_model.eval()

with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
text_tokens = clip.tokenize(categories).to(device)


def predict_clip(image_path):
    image = clip_preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_prob, top_idx = similarity[0].topk(1)

    label = categories[top_idx.item()]
    prob = top_prob.item()
    return label, prob


filenames = glob.glob(args.input)
filenames.sort()
total_images = len(filenames)
print(f"Total images: {total_images}")


with open(args.output, 'w') as f:
    f.write("Image\tObject\tProbability\n")

for image_path in filenames:
    label, prob = predict_clip(image_path)
    image_name = os.path.basename(image_path)
    print(f"{image_name}: {label} ({prob:.4f})")

    with open(args.output, 'a') as f:
        f.write(f"{image_name}\t{label}\t{prob:.4f}\n")

print(f"\nResults saved to '{args.output}'")
