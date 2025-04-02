import os
import pandas as pd
import numpy as np
from PIL import Image

from clahe import claheimg
import torch
import cv2
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split,Dataset
import torch
from torchvision import transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt

print(len(os.listdir("rep/images")))

class FingerprintData(Dataset):
    
    def __init__(self, image_dir, label_dir, transform = None, sigma = 2.0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.sigma = sigma
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg'))]
        self.label_files = [f.replace(os.path.splitext(f)[1], '.tsv') for f in self.image_files]
        for lf in self.label_files:
            assert os.path.isfile(os.path.join(label_dir, lf)), f"Missing: {lf}"
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = self.label_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        #aqui chama-se a função clahe
        image = Image.fromarray(claheimg(img_path))
        
        label_path = os.path.join(self.label_dir, label_name)
        coords = pd.read_csv(label_path, sep = '\t', names = ['x', 'y']).values
        
        originw, originh = image.size
        resized_image = image
        
        #resized_coords = self._resize_coords(coords[1:], (originw, originh), (300,300))




        
        if self.transform:
            image = self.transform(resized_image)
        heatmap = self._create_heatmap(coords, (256,256))
        heatmap = torch.tensor(heatmap, dtype = torch.float32).unsqueeze(0)
        
        
        return image, heatmap

    def _resize_coords(self, coords, original_size, target_size):
        originw, originh = original_size
        tarw, tarh = target_size
        
        scale_x = tarw / originw
        scale_y = tarh / originh
       
        coords = np.array(coords, dtype=np.float32)
        resized_coords = coords * np.array([scale_x, scale_y])

        return resized_coords
    def _create_heatmap(self, coords, img_size):
        heatmap = np.zeros(img_size, dtype = np.float32)
        h, w = img_size
        window_size = int(6*self.sigma) + 1
        radius = window_size // 2
        coords = coords[1:]
        for x, y in coords:
            x, y = float(x), float(y)
            x_min = max(0, int(x - radius))
            x_max = min(w, int(x + radius + 1))
            y_min = max(0, int(y - radius))
            y_max = min(h, int(y + radius + 1))
            
            if x_min >= x_max or y_min >= y_max:
                continue
            
            xx, yy = np.meshgrid(np.arange(x_min, x_max), np.arange(y_min, y_max))
            gaussian = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * self.sigma**2))
            heatmap[y_min:y_max, x_min:x_max] += gaussian

        heatmap = np.clip(heatmap, 0, 1)
        return heatmap
    





def show_images(image1, image2, image3, filename):
    filename = filename + ".png"
    if image3 is None:
        image1_np = image1.squeeze().numpy()
        image2_np = image2.squeeze().numpy()
        image1_np = (image1_np * 255).astype(np.uint8)
        image2_np = (image2_np * 255).astype(np.uint8)

        if image1_np.shape != image2_np.shape:
            h = min(image1_np.shape[0], image2_np.shape[0])
            w1 = int((h / image1_np.shape[0]) * image1_np.shape[1])
            w2 = int((h / image2_np.shape[0]) * image2_np.shape[1])
            image1_np = cv2.resize(image1_np, (w1, h))
            image2_np = cv2.resize(image2_np, (w2, h))
        _, axs = plt.subplots(1, 2, figsize=(12, 12))
        axs = axs.flatten()
        axs[0].imshow(image1_np)
        axs[0].set_title('Imagem 1')
        axs[1].imshow(image2_np)
        axs[1].set_title('Imagem 2')

        
        for ax in axs:
            ax.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    else:
        image1_np = image1.squeeze().numpy()
        image2_np = image2.squeeze().numpy()
        image3_np = image3
        image1_np = (image1_np * 255).astype(np.uint8)
        image2_np = (image2_np * 255).astype(np.uint8)
        image3_np = (image3_np * 255).astype(np.uint8)

        if image1_np.shape != image2_np.shape:
            h = min(image1_np.shape[0], image2_np.shape[0])
            w1 = int((h / image1_np.shape[0]) * image1_np.shape[1])
            w2 = int((h / image2_np.shape[0]) * image2_np.shape[1])
            w3 = int((h / image3_np.shape[0]) * image3_np.shape[1])
            image1_np = cv2.resize(image1_np, (w1, h))
            image2_np = cv2.resize(image2_np, (w2, h))
            image3_np = cv2.resize(image3_np, (w3, h))
       
        _, axs = plt.subplots(1, 3, figsize=(12, 12))
        axs = axs.flatten()
        axs[0].imshow(image1_np, cmap = 'gray')
        axs[0].set_title('Imagem Real')
        axs[1].imshow(image2_np, cmap = 'gray')
        axs[1].set_title('Imagem Label')

        axs[2].imshow(image3_np, cmap = 'gray')
        axs[2].set_title('Imagem do Modelo')
        for ax in axs:
            ax.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)


