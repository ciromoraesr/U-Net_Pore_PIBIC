from process import FingerprintData, show_images
import os
import architecture
import architecture2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import random
from datetime import datetime
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.transforms import v2




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

torch.cuda.empty_cache()

v = input("Wanna train the model? [Y/n]")
if(v.lower() == 'y'):
    h = architecture2.train_model(train_loader, val_loader, train_size, val_size, date_today)




model = architecture2.EnhancedPoreDetectionCNN()




def plot_train(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    time = datetime.now()
    
    pos_weight = torch.tensor([10.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion.to(device)
    checkpoint = torch.load("model_folder/best_enhanced_model_01-04-2025.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    t_loss, t_acc, t_metrics = architecture2.evaluate(model, test_loader, test_size, device,criterion)
    
    print(t_loss, t_acc)
    for i in range(2):
        file = str(i) + "_" + str(time.hour)
        filename = os.path.join('prints/', file)
        random_idx = random.randint(0, len(test_dataset) - 1)
        image_ex, heat_ex = test_dataset[random_idx]
        image = image_ex.unsqueeze(0).to(device)
    
        model.eval()

        with torch.no_grad():
            output = model(image)
        
        output = output.squeeze(0).cpu().numpy()
        output = output.squeeze(0)
        show_images(image_ex, heat_ex, output, filename)
    return t_loss, t_acc, t_metrics


def plot(model, save_path="training_plot_", date_today=""):
   
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))
    
    
    axs[0].plot(model['history']['train_acc'], color="red", marker="o")
    axs[0].plot(model['history']['val_acc'], color="blue", marker="h")
    axs[0].set_title('Accuracy Comparison between Train & Validation Set')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(['Train', 'Validation'], loc="lower right")
    
   
    axs[1].plot(model['history']['train_loss'], color="red", marker="o")
    axs[1].plot(model['history']['val_loss'], color="blue", marker="h")
    axs[1].set_title('Loss Comparison between Train & Validation Set')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(['Train', 'Validation'], loc="upper right")

    axs[2].plot(model['history']['val_iou'], color="blue", marker="h")
    axs[2].set_title('IoU from Validation Set')
    axs[2].set_ylabel('Iou')
    axs[2].set_xlabel('Epoch')
    axs[2].legend('Validation', loc="upper right")

    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {save_path+date_today+'.png'}")
    


train_loss, train_acc, t_metrics = plot_train(model)
print(train_loss,"e acurácia de treino:", train_acc)


size_histories = {}


try:

    size_histories['Model'] = {'history':h}
    
    plot(size_histories['Model'], date_today)
except:
    print("O modelo não foi treinado")

    




