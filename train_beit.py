import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from classification_model import BEiTImageClassifier, MobileViTImageClassifier
from dataset import ImageDataset
from global_variables import *
from torch.utils.data import DataLoader, random_split

def resize_image(image_tensor, new_height, new_width):
    # image_tensor shape should be (B, C, H, W) or (C, H, W)
    if len(image_tensor.shape) == 3:
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension if it's not present
    
    resized_tensor = F.interpolate(image_tensor, size=(new_height, new_width), mode='bilinear', align_corners=False)
    return resized_tensor

def train(model, train_dataloader, validation_dataloader = None):
    # Define the loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    num_epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    for epoch in range(num_epochs):
        # torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        train_dataset = ImageDataset(TRAIN_CSV_PATH, IMG_DIR)
        train_dataloader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=2)
        for images, labels, _, _, _ in train_dataloader:
            images, labels = images.to(device), labels.to(device)
            images = resize_image(images.clone().detach(), 224, 224)
            # Forward pass
            outputs = model(images)#.squeeze(-1)
            loss = criterion(outputs, labels.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Compute training loss and accuracy
            train_loss += loss.item() * images.size(0)
            predicted_labels = (torch.sigmoid(outputs) > 0.5).int()
            total_train += labels.size(0)
            correct_train += (predicted_labels == labels).sum().item()

        train_loss /= total_train
        train_accuracy = 100 * correct_train / total_train


        torch.save(model.state_dict(), f'beit_model_{epoch}.pth')

        # Evaluate the model on the validation set
        validation_loss = 0.0
        correct_validation = 0
        total_validation = 0
        validation_accuracy = 0

        if validation_dataloader:
            model.eval()
            with torch.no_grad():
                for images, labels, _, _, _ in validation_dataloader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)#.squeeze(-1)
                    loss = criterion(outputs, labels.float())

                    validation_loss += loss.item() * images.size(0)
                    predicted_labels = (torch.sigmoid(outputs) > 0.5).int()
                    total_validation += labels.size(0)
                    correct_validation += (predicted_labels == labels).sum().item()

            validation_loss /= total_validation
            validation_accuracy = 100 * correct_validation / total_validation

        print(f'Epoch [{epoch + 1}/{num_epochs}]: '
            f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, ')

if __name__=='__main__':
    model = BEiTImageClassifier(num_classes=1)
    b = 64
    validation_test = False
    if validation_test:
        dataset = ImageDataset(TRAIN_CSV_PATH, IMG_DIR)

        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=b, shuffle=False, num_workers=2)
        test_dataloader = DataLoader(test_dataset, batch_size=b, shuffle=False, num_workers=2)

        train(model, train_dataloader, test_dataloader)
    else:
        train_dataset = ImageDataset(TRAIN_CSV_PATH, IMG_DIR)
        train_dataloader = DataLoader(train_dataset, batch_size=b, shuffle=True, num_workers=2)

        train(model, train_dataloader)


