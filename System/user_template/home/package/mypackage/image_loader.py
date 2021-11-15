import os
import torch
from torchvision import transforms
from PIL import Image

def load_image(paths):
    out = []
    lab = []
    for path in paths:
        img = Image.open(path) 
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28)),
            transforms.ToTensor()
        ])
        out.append(transform(img).unsqueeze(0))
        lab.append(os.path.basename(path))
    return torch.cat(out, dim=0), lab