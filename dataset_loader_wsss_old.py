import cv2
from torchvision import  transforms
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import torch
import logging
logging.getLogger('PIL').setLevel(logging.INFO)

# DARE statistics
# mean = (0.5298, 0.4849, 0.4408)
# std = (0.1864, 0.1896, 0.1933)
#
#
# normalize = transforms.Normalize(mean=mean, std=std)
#
# train_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomApply([
#         transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
#     ], p=0.6),
#     transforms.RandomGrayscale(p=0.2),
#     transforms.ToTensor(),
#     normalize,
# ])
import numpy as np
class ToTensorNoNormalize:
    def __call__(self, pic):
        # Convert PIL image to NumPy array
        np_img = np.array(pic, dtype=np.uint8)
        # Convert NumPy array to a PyTorch tensor
        return torch.from_numpy(np_img).long()
class SupConDataset(Dataset):

    def __init__(self, root_dir,
                 mode='train',
                 annot_dir="masks",
                 img_dir="imgs",
                 transform=None
                 ):

        self.root_dir = root_dir
        self.mode = mode

        self.imgs_dir = os.path.join(root_dir, img_dir, mode).replace('\\', '/')
        self.labels_dir = os.path.join(root_dir, annot_dir, mode).replace('\\', '/')

        self.list_of_images = os.path.join(root_dir, f'dare_{mode}_s_c.txt').replace('\\', '/')

        self.get_list_of_dataset()

        self.transform = transform


    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def correct_exif_rotation(img):
        try:
            exif = img.getexif()
            if not exif:
                return img  # No EXIF, return original image

            orientation = exif.get(0x0112)  # 0x0112 is the EXIF tag for orientation
            if orientation == 3:
                img = img.rotate(180, expand=True)
            elif orientation == 6:
                img = img.rotate(270, expand=True)
            elif orientation == 8:
                img = img.rotate(90, expand=True)
        except AttributeError:
            pass  # No EXIF data

        return img

    def __getitem__(self, index):
        _labels = torch.zeros(3)
        seg_img_name, lbl = self.dataset[index].strip().split(',')

        # _valid_mask = torch.zeros((3, 28, 28))

        orig_img_name = f'{seg_img_name.strip().split("_")[0]}_img.png'  # 10052_img.png

        img = Image.open(os.path.join(self.imgs_dir, orig_img_name), mode='r').convert('RGB')
        img = self.correct_exif_rotation(img)

        # mask = Image.open(os.path.join('../copied_focal/dataset/masks/train', seg_img_name), mode='r').convert('L')
        # mask = cv2.imread(os.path.join(self.labels_dir, seg_img_name), cv2.IMREAD_GRAYSCALE)

        # mask = transforms.Compose([
        #     transforms.Resize((56, 56), interpolation=Image.NEAREST),  # Nearest neighbor interpolation to keep labels intact
        #     ToTensorNoNormalize()  # Use custom transformation to avoid normalization
        #         ])(mask)

        img = self.transform(img)

        _labels[int(lbl)-1] = 1
        # _valid_mask[int(lbl)-1] = 1

        valid_mask = torch.zeros((4, 224, 224))
        valid_mask[int(lbl)] = 1
        labels = torch.zeros(3)
        labels[int(lbl)-1] = 1





        # return img, int(lbl)-1, str(os.path.join(self.imgs_dir, orig_img_name))
        # return img, _labels, str(os.path.join(self.imgs_dir, orig_img_name)), _valid_mask
        return {
            'images' : img,
            'name' : str(os.path.join(self.imgs_dir, orig_img_name)),
            'labels' : labels,
            'labels_1' : int(lbl)-1,
            'valid_mask' : valid_mask,
            'cat': int(lbl)
        }




    def get_list_of_dataset(self):
        lines_list = []

        with open(self.list_of_images, 'r') as file:
            for line in file:
                lines_list.append(line.strip())

        self.dataset = lines_list



