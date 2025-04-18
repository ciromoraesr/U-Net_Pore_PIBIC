from process import FingerprintData
import os
import architecture
import architecture2
import torch
import torch.nn as nn
import torch.optim as optim
import random
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import v2
from tests import plot_results, test_acc, plot_example, overlay

torch.cuda.empty_cache()

exform = transforms.Compose([
    transforms.ToTensor(),
   
])

#setting a seed to fix the test, train and validation data
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

img_dir = r"/home/cirorocha/PIBIC_CIRO/repository/new_images/"
lbl_dir = r"/home/cirorocha/PIBIC_CIRO/repository/new_labels/"
dataset = FingerprintData(img_dir, lbl_dir, transform = exform)


#spliting the dataset into 80% train, 10% validation and 10% test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


date_today = datetime.now().strftime("%d-%m-%Y-%H-%M")


v = input("Wanna train the model? [Y/n]")

if(v.lower() == 'y'):
    h = architecture2.train_model(train_loader, val_loader, train_size, val_size, date_today, num_epochs = 60)
    
   
    savepath = os.path.join("/home/cirorocha/PIBIC_CIRO/results/", "training_plot_"+date_today)
    plot_results(h, savepath)

model_1 = "/home/cirorocha/PIBIC_CIRO/model_folder/best_model_16-04-2025-08-26.pth"
model_2 = "/home/cirorocha/PIBIC_CIRO/model_folder/best_model_15-04-2025-23-09.pth"

model1 = architecture2.EnhancedPoreDetectionCNN()
model2 = architecture2.PoreDetectionCNN2()
# plot_example(model, model_now, test_dataset)
# cont, coords = overlay(model, model_now, test_dataset)

print("modelo u net")
test_loss, test_accuracy, t_metrics = test_acc(model1,model_1, test_loader, test_size)
print(test_accuracy,t_metrics)
print("modelo resnet")
test_loss, test_accuracy, t_metrics = test_acc(model2,model_2, test_loader, test_size)
print(test_accuracy,t_metrics)



    




