from process import FingerprintData, show_images
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
from tests import plot_results, test_acc, plot

torch.cuda.empty_cache()

exform = transforms.Compose([
    transforms.ToTensor(),
   
])

img_dir = r'repository/new_images/'
lbl_dir = r'repository/new_labels/'
dataset = FingerprintData(img_dir, lbl_dir, transform = exform)


#spliting the dataset into 80% train, 10% validation and 10% test
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


date_today = datetime.now().strftime("%d-%m-%Y-%H")


v = input("Wanna train the model? [Y/n]")

if(v.lower() == 'y'):
    h = architecture2.train_model(train_loader, val_loader, train_size, val_size, date_today, 20)


model = architecture2.EnhancedPoreDetectionCNN()


test_loss, test_acc, t_metrics = test_acc(model)
print(test_loss,"e acurácia de treino:", test_acc)

size_histories = {}

try:
    size_histories['Model'] = {'history':h}
    savepath = os.path.join(r"/results", "training_plot_"+date_today)
    plot_results(size_histories['Model'], )
except:
    print("O modelo não foi treinado")

    




