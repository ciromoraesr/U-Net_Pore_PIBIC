import torch
import architecture2
import random
import os
from datetime import datetime
from process import FingerprintData, show_images
from torchvision import transforms
import time
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


def plot(model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    

    checkpoint = torch.load("model_folder/best_enhanced_model_01-04-2025.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    
   
    file = "test_" + datetime.now().strftime("%d-%m-%H-%M")
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
    print(f"test image saved at {filename}")


def test_acc(model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    
    criterion = architecture2.CombinedLoss(bce_weight=0.5).to(device)

    checkpoint = torch.load("model_folder/best_enhanced_model_01-04-2025.pth")

    model.load_state_dict(checkpoint['model_state_dict'])

    t_loss, t_acc, t_metrics = architecture2.evaluate(model, test_loader, test_size, device,criterion)
    
    
    return t_loss, t_acc, t_metrics





def plot_results(history, save_path):
    
    
    
    plt.figure(figsize=(12, 10))
    
    epochs = range(1, len(history['train_acc']) + 1) 
    
    plt.subplot(3, 1, 1)
    plt.plot(epochs, history['train_acc'], 'r-', label='Train')  
    plt.plot(epochs, history['val_acc'], 'b-', label='Validation')
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)  
    plt.legend()
    
    
    plt.subplot(3, 1, 2)
    plt.plot(epochs, history['train_loss'], 'r-', label='Train')
    plt.plot(epochs, history['val_loss'], 'b-', label='Validation')
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xticks(epochs)  
    plt.legend()
    
    
    plt.subplot(3, 1, 3)
    if 'val_iou' in history:
        plt.plot(epochs, history['val_iou'], 'g-', label='Validation IoU')
        plt.title('IoU')
        plt.ylabel('IoU Score')
        plt.xticks(epochs)  
        plt.legend()
    
    plt.xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


exform = transforms.Compose([
    transforms.ToTensor(),
   
])


img_dir = r'rep/images'
lbl_dir = r'rep/labels'

dataset = FingerprintData(img_dir, lbl_dir, transform = exform)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])




test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


model = architecture2.EnhancedPoreDetectionCNN()




