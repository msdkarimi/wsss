import os.path
from  pathlib import Path
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import cv2
from utils import load_yaml
from dataset_loader_sup_con_700_clean_3_images_cam_to_label import SupConDataset_WSSS
import torchvision.transforms as transforms
import torch
import numpy as np
from backbone_wsss import FocalNet
from torch.utils.data import DataLoader
from model_util import load_weights
from SepervisedContrastive_model_wsss_used_used_to_get_cls_cam_compact import SupCon_CAM
from SepervisedContrastive_model_wsss_grad_cam_to_label import SupCon_WSSS
import matplotlib.pyplot as plt
from collections import defaultdict
import json
from sklearn.metrics import jaccard_score


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # normalize,
])
_MAPPING = {0:0, 1:1, 3:2}
_MAPPING_COLOR =     {  0 : (0, 0, 255), # "Bright Red"
                        1 : (0, 255, 0), #"Bright Green"
                        2 : (255, 0, 0),} #"Yellow"
#
# {0: (255, 0, 0),  # "Bright Red"
#  1: (0, 255, 0),  # "Bright Green"
#  2: (255, 255, 0), }  # "Yellow"


def convert_cam_to_labels(dataloader, model):

    model.eval()

    _ious = defaultdict(list)
    _names = defaultdict(list)

    for idx, (samples, targets, names, masks) in enumerate(dataloader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        act, cams, _, _ = model(samples, targets, names)

        for idx in range(act.shape[0]):
            # img = np.array(samples[idx].detach().cpu())
            # img = cv2.imread(os.path.join( '../', ".".join(names[idx].split('.')[2:])))
            img = cv2.imread(names[idx])
            msk = cv2.imread(masks[idx], cv2.IMREAD_GRAYSCALE)
            _lbl = targets[idx]

            binary_mask = np.where(msk > 0, 1, 0)

            # # Method 2: Boolean indexing (also efficient)
            # binary_mask = msk.copy()  # Important: Create a copy to avoid modifying the original mask
            # binary_mask[binary_mask > 0] = 1







            h, w = msk.shape
            _cam = cams[idx].detach().cpu()
            _activation = act[idx].detach().cpu()

            _activation_1 = (4 * (_activation / 5)) + (1*(_cam/5))
            _min = _activation_1.min()
            _max = _activation_1.max()
            _activation_2 = (_activation_1 - _min)/ (_max - _min)
            am = torch.nn.functional.interpolate(_activation_2.unsqueeze(0).unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
            am = am[0][0].detach().cpu().numpy()
            _all = np.zeros((3, h, w))
            _all[_lbl.item(), :, :] = am



            fg_conf_cam = np.pad(_all, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.75)
            fg_conf_cam = np.argmax(fg_conf_cam, axis=0)

            d = dcrf.DenseCRF2D(w, h, 4)
            unary = unary_from_labels(fg_conf_cam, 4, gt_prob=0.99, zero_unsure=False)
            d.setUnaryEnergy(unary)
            d.addPairwiseGaussian(sxy=3, compat=3)
            d.addPairwiseBilateral(sxy=100, srgb=50, rgbim=np.ascontiguousarray(np.copy(img)), compat=11)
            q = d.inference(10)
            r = np.argmax(np.array(q).reshape((4, h, w)), axis=0)

            label_file_name = names[idx].strip().split('/')[-1].split('.')[0]

            label = np.where(r > 0, 1, 0)

            cv2.imwrite(f'labels/{label_file_name}.png', r)
            rgb_mask = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
            rgb_mask[label == 1] = _MAPPING_COLOR[_lbl.item()]
            cv2.imwrite(f'colo_labels/{label_file_name}.png', rgb_mask)

            # plt.imshow(r)
            # plt.show()
            # plt.imshow(msk)
            # plt.show()
            iou = jaccard_score(binary_mask.flatten(), label.flatten())
            _ious[str(_lbl.item())].append(iou.item())
            _names[str(_lbl.item())].append(names[idx])

        print(_ious)
        print(_names)

    with open("val_ious.json", 'w') as f:
        json.dump(_ious, f, indent=4)

    with open("val_names.json", 'w') as f:
        json.dump(_names, f, indent=4)

    print('msd')




        # img = cv2.imread(r'C:\Users\massoud\PycharmProjects\700perizie_cleaning\wsss_image_annotation_extaction\images\7022.jpg')
        # h, w = img.shape[:2]
        # _cam = torch.from_numpy(grayscale_cam).cuda()
        # _activation_1 = (4 * (_activation / 5)) + (1*(_cam/5))
        # _min = _activation_1.min()
        # _max = _activation_1.max()
        # _activation_2 = (_activation_1 - _min)/ (_max - _min)
        # am = torch.nn.functional.interpolate(_activation_2.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False)
        # am = am[0][0].detach().cpu().numpy()
        # _all = np.zeros((3, h, w))
        # _all[0, :, :] = am
        # fg_conf_cam = np.pad(_all, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.80)
        # fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
        # d = dcrf.DenseCRF2D(w, h, 4)
        # unary = unary_from_labels(fg_conf_cam, 4, gt_prob=0.99, zero_unsure=False)
        # d.setUnaryEnergy(unary)
        # d.addPairwiseGaussian(sxy=3, compat=3)
        # d.addPairwiseBilateral(sxy=100, srgb=50, rgbim=np.ascontiguousarray(np.copy(img)), compat=11)
        # q = d.inference(20)
        # r = np.argmax(np.array(q).reshape((4, h, w)), axis=0)
        # plt.imshow(r)
        # plt.show()

if __name__ == '__main__':
    yaml_data = load_yaml('config_cls.yml')
    dataset_train = SupConDataset_WSSS(root="../cleaning/dataset/wsss", mode="val", transform=val_transform, label_mapping={0: 0, 1: 1, 3: 2})
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)

    focalnets = FocalNet( depths=yaml_data["MODEL"]["FOCAL"]["DEPTHS"],
                      embed_dim=yaml_data["MODEL"]["FOCAL"]["EMBED_DIM"],
                      focal_levels=yaml_data["MODEL"]["FOCAL"]["FOCAL_LEVELS"],
                      focal_windows=yaml_data["MODEL"]["FOCAL"]["FOCAL_WINDOWS"],
                      drop_path_rate=yaml_data["MODEL"]["FOCAL"]["DROP_PATH_RATE"],
                      num_classes= int(yaml_data["MODEL"]["FOCAL"]["NUM_CLASSES"]),)

    cam_model = SupCon_CAM(encoder=focalnets, head='cls', input_feature_dim=768, output_feature_dim=768, num_classes=3)
    cam_model = cam_model.cuda()
    # cam_model, _, _ = load_weights(cam_model, "pretrained_dare/3_images_w_nll_cls_0.89083.pth", mine=True)
    model = SupCon_WSSS(encoder_cam=cam_model)
    model, _, _ = load_weights(model, "pretrained_dare/cam_checkpoint_1.pth", mine=True)
    model = model.cuda()

    convert_cam_to_labels(train_loader, model)
    exit(0)
