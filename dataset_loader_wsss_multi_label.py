import cv2
from torchvision import  transforms
from torch.utils.data import Dataset
import os
from PIL import Image, ImageOps
import torch
import logging
logging.getLogger('PIL').setLevel(logging.INFO)
import ast

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

MAPPING = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 10: 8, 11: 9, 12: 10, 14: 11}

class SupConDataset(Dataset):

    def __init__(self, root_dir,
                 mode='train',
                 annot_dir="masks",
                 img_dir="images",
                 transform=None
                 ):

        self.root_dir = root_dir
        self.mode = mode

        self.imgs_dir = os.path.join(root_dir, img_dir).replace('\\', '/')
        self.labels_dir = os.path.join(root_dir, annot_dir, 'semantic').replace('\\', '/')

        self.list_of_images = os.path.join(root_dir, f'{mode}.txt').replace('\\', '/')

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
        seg_img_name, lbl, _multi_label = self.dataset[index].strip().split(':')
        _multi_label = ast.literal_eval(_multi_label)
        _present_labels =  ast.literal_eval(lbl)
        # _valid_mask = torch.zeros((3, 28, 28))

        # orig_img_name = f'{seg_img_name.strip().split("_")[0]}_img.png'  # 10052_img.png
        orig_img_name = f'{seg_img_name}.jpg'  # 10052_img.png
        mask_name = f'{seg_img_name}.png'
        _mask_dir = os.path.join(self.labels_dir, mask_name)
        msk = Image.open(os.path.join(self.labels_dir, mask_name), mode='r').convert('L')

        _image_name = os.path.join(self.imgs_dir, orig_img_name)

        img = Image.open(os.path.join(self.imgs_dir, orig_img_name), mode='r').convert('RGB')
        img = ImageOps.exif_transpose(img)
        # img = self.correct_exif_rotation(img)

        mask = transforms.Compose([
            transforms.Resize((224, 224), interpolation=Image.NEAREST),  # Nearest neighbor interpolation to keep labels intact
            ToTensorNoNormalize()  # Use custom transformation to avoid normalization
                ])(msk)

        img = self.transform(img)
        # _lbl = 2 if int(lbl) == 3 else int(lbl)
        # _labels[_lbl] = 1
        # _valid_mask[int(lbl)-1] = 1

        valid_mask = torch.zeros((13, 224, 224))
        # _lbl_ = int(lbl) + 1 if int(lbl) <= 2 else int(lbl)
        for _pres_label in _present_labels:
            if _pres_label in MAPPING:
                valid_mask[MAPPING[_pres_label] + 1] = 1
        # labels = torch.zeros(3)
        # labels[_lbl] = 1



        _cats = [MAPPING[item] for item in _present_labels if item in MAPPING.keys()]

        while len(_cats) < 14:
            _cats.append(-1)


        # return img, int(lbl)-1, str(os.path.join(self.imgs_dir, orig_img_name))
        # return img, _labels, str(os.path.join(self.imgs_dir, orig_img_name)), _valid_mask
        return {
            'images' : img,
            'name' : str(_mask_dir),
            'img_name' : str(_image_name),
            'labels' : torch.from_numpy(np.array(_multi_label)),
            # 'labels_1' : int(lbl)-1,
            'valid_mask' : valid_mask,
            'cat': torch.from_numpy(np.array(_cats)) + 1,
            'gt_mask': mask,
            'image_number': seg_img_name
        }




    def get_list_of_dataset(self):
        lines_list = []

        with open(self.list_of_images, 'r') as file:
            for line in file:
                lines_list.append(line.strip())

        self.dataset = lines_list



