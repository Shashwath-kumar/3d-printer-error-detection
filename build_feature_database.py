import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ImageDataset
from edge_detection import LDC
from global_variables import *
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import kornia as kn
from PIL import Image

def image_normalization(img, img_min=0, img_max=255,
                        epsilon=1e-12):
    """This is a typical image normalization function
    where the minimum and maximum of the image is needed
    source: https://en.wikipedia.org/wiki/Normalization_(image_processing)

    :param img: an image could be gray scale or color
    :param img_min:  for default is 0
    :param img_max: for default is 255

    :return: a normalized image, if max is 255 the dtype is uint8
    """

    img = np.float32(img)
    # whenever an inconsistent image
    img = (img - np.min(img)) * (img_max - img_min) / \
        ((np.max(img) - np.min(img)) + epsilon) + img_min
    return img

def save_image_batch_to_disk(tensor, file_names, img_dirs, img_shape=[torch.tensor([IMG_HEIGHT]), torch.tensor([IMG_WIDTH])]):

    
    output_dir_f = os.path.join(LDC_OUTPUT_DIR, LDC_IMAGE_FOLDER)
    os.makedirs(output_dir_f, exist_ok=True)

    # 255.0 * (1.0 - em_a)
    edge_maps = []
    for i in tensor:
        tmp = torch.sigmoid(i).cpu().detach().numpy()
        edge_maps.append(tmp)
    tensor = np.array(edge_maps)
    # print(f"tensor shape: {tensor.shape}")
    # print(f"img_shape shape: {img_shape}")

    image_shape = [x.cpu().detach().numpy() for x in img_shape]
    image_shape = [[y, x] for x, y in zip(image_shape[0], image_shape[1])]

    assert len(image_shape) == len(file_names)

    idx = 0
    for i_shape, file_name, img_dir in zip(image_shape, file_names, img_dirs):
        tmp = tensor[:, idx, ...]
        tmp = np.squeeze(tmp)

        # Iterate our all 7 NN outputs for a particular image
        preds = []
        fuse_num = tmp.shape[0]-1
        for i in range(tmp.shape[0]):
            tmp_img = tmp[i]
            tmp_img = np.uint8(image_normalization(tmp_img))
            tmp_img = cv2.bitwise_not(tmp_img)

            # Resize prediction to match input image size
            if not tmp_img.shape[1] == i_shape[0] or not tmp_img.shape[0] == i_shape[1]:
                tmp_img = cv2.resize(tmp_img, (i_shape[0], i_shape[1]))
            
            preds.append(tmp_img)

            if i == fuse_num:
                fuse = tmp_img
                fuse = fuse.astype(np.uint8)

        output_file_dir = output_dir_f
        for dir in img_dir.split('/'):
            output_file_dir = os.path.join(output_file_dir, dir)
            os.makedirs(output_file_dir, exist_ok=True)
        output_file_name_f = os.path.join(output_file_dir, file_name)
        cv2.imwrite(output_file_name_f, fuse)
        idx += 1

def create_LDC_images(dataloader, model, device, train= True):

    model.eval()    
    with torch.no_grad():
        for batch_id, sample_batched in enumerate(dataloader):
            images = sample_batched[0].to(device)
            file_names = sample_batched[2]
            img_dirs = sample_batched[3]
            preds = model(images)
            save_image_batch_to_disk(preds, file_names, img_dirs)


def main():
    device = torch.device('cpu' if torch.cuda.device_count() == 0
                          else 'cuda')
    print(device)
    LDC_model = LDC().to(device)
    LDC_model.load_state_dict(torch.load(LDC_CHECKPOINT_PATH, map_location=device))

    train_dataset = ImageDataset(TRAIN_CSV_PATH, IMG_DIR)

    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

    create_LDC_images(data_loader, LDC_model, device)

    train_dataset = ImageDataset(TEST_CSV_PATH, IMG_DIR)

    data_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=2)

    create_LDC_images(data_loader, LDC_model, device)


if __name__=='__main__':    
    main()