import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from classification_model import ImageClassificationModel, BEiTImageClassifier, MobileViTImageClassifier
from dataset import ImageDataset
from global_variables import *
from torch.utils.data import DataLoader
import pandas as pd

def resize_image(image_tensor, new_height, new_width):
    # image_tensor shape should be (B, C, H, W) or (C, H, W)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension if it's not present
    
    resized_tensor = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized_tensor

def train(model, test_dataloader):
    # Define the loss function and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    df = pd.DataFrame(columns=['img_path', 'has_under_extrusion'])
    model.eval()
    idx = 0
    with torch.no_grad():
        for images, labels, _, _, img_paths in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            # images = resize_image(images.clone().detach(), 224, 224)
            outputs = model(images)
            
            predicted_labels = (torch.sigmoid(outputs) > 0.5).int()
            predicted_labels = np.array(predicted_labels.to('cpu'), dtype=int)
            df = pd.concat([df, pd.DataFrame({'img_path':img_paths, 'has_under_extrusion':predicted_labels})])
            idx += len(img_paths)
            print(idx)
    df.to_csv('submission_2.csv', index=False)

if __name__=='__main__':
    model = ImageClassificationModel(num_classes=1)
    model.load_state_dict(torch.load('model_7.pth'))

    b = 64
    test_dataset = ImageDataset(TEST_CSV_PATH, IMG_DIR)

    test_dataloader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=2) 

    train(model, test_dataloader)