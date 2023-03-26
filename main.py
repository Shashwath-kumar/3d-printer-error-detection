import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import TrainImageDataset
from edge_detection import LDC
from global_variables import *
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

def test_LDC(dataloader, model, device):

    model.eval()
    
    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched[0].to(device)
            print(images.shape)
            label = sample_batched[1]
            preds = model(images)
            img_np = preds[0][0][0].numpy()
            plt.imshow(img_np, cmap='gray')
            plt.show()
            break


def main():
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    LDC_model = LDC().to(device)
    LDC_model.load_state_dict(torch.load(LDC_CHECKPOINT_PATH, map_location=device))

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(IMG_HEIGHT, IMG_WIDTH))
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


    train_dataset = TrainImageDataset(TRAIN_CSV_PATH, IMG_DIR, transform=transform)

    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)


    test_LDC(data_loader, LDC_model, device)

if __name__=='__main__':    
    main()