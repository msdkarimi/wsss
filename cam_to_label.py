import numpy as np
import torch.nn.functional as F
# from model import Net
from model_cpfe import Net
from dataset_loader_wsss import SupConDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
from utils import PolyOptimizerSGD, AverageMeter, convert_image, generate_vis, VOClabel2colormap
from torch.utils.tensorboard import SummaryWriter
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
import matplotlib.pyplot as plt


def crf_inference_label(img, labels, t=10, n_labels=21, gt_prob=0.3):

    _, h, w = img.shape
    img = img.transpose(1, 2, 0)

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=1, compat=2)
    # d.addPairwiseBilateral(sxy=10, srgb=5, rgbim=np.ascontiguousarray(np.copy(img)), compat=10)
    d.addPairwiseBilateral(sxy=5, srgb=3, rgbim=np.ascontiguousarray(np.copy(img)), compat=5)

    q = d.inference(t)

    return np.argmax(np.array(q).reshape((n_labels, h, w)), axis=0)


def compute_iou_segmentation(mask1, mask2):
    # Ensure masks are binary (0s and 1s) or convert them
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)

    # Intersection and Union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    # IoU calculation
    iou = intersection / union if union > 0 else 0
    return iou

if __name__ == '__main__':
    model = Net(num_cls=4)
    model.load_state_dict(torch.load('check_more_prototype/model_12001.pth'))
    # model.load_state_dict(torch.load('checkpoints/model_12001.pth'))
    model = model.cuda()

    model.eval()
    epochs = 1
    print_freq = 10
    writer = SummaryWriter('logs/')

    # train_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\copied_focal\copied_focal\dataset',
    train_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction',
                                  mode='train',
                                  transform=transforms.Compose(
                                      [
                                          transforms.Resize((224, 224)),
                                          transforms.RandomGrayscale(),
                                          transforms.ToTensor(),
                                      ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=512, shuffle=True)

    val_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction',
                                  mode='val',
                                  transform=transforms.Compose(
                                      [
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                      ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=256, shuffle=False)

    max_step = epochs*len(train_loader)

    ious = {'0': [],
               '1': [],
               '3': []
               }
    names = {'0': [],
               '1': [],
               '3': []
               }

    with torch.no_grad():
        for batch_idx, pack in enumerate(train_loader):
            img = pack['images'].cuda()
            n, c, h, w = img.shape
            label = pack['labels'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)

            valid_mask = pack['valid_mask'].cuda()
            valid_mask[:,1:] = valid_mask[:,1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(h // 16, w // 16), mode='nearest')
            outputs = model.forward(img, valid_mask_lowres, name=pack['name'], category=pack['cat'])

            gt_mask  = pack['gt_mask']
            cats = pack['cat']
            _names = pack['name']

            score = outputs['score']
            norm_cam = outputs['cam']
            IS_cam = outputs['IS_cam']

            lossCLS = F.multilabel_soft_margin_loss(score, label, weight=torch.tensor([0.72297, 0.687215, 0.589804]).cuda())

            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            lossGSC = torch.mean(torch.abs(norm_cam - IS_cam)) * 2

            for _b_idx in range(score.shape[0]):
                # _norm_cam = norm_cam[_b_idx]
                # _IS_cam = IS_cam[_b_idx]
                name = _names[_b_idx]
                _cat = cats[_b_idx]
                _gt_mask = gt_mask[_b_idx].numpy()
                _img = convert_image(img[_b_idx])
                _norm_cam_ = F.interpolate(norm_cam, _img.shape[1:], mode='bilinear')[_b_idx].detach().cpu().numpy()
                IS_cam_ = F.interpolate(IS_cam, _img.shape[1:], mode='bilinear')[_b_idx].detach().cpu().numpy()
                CAM = generate_vis(_norm_cam_, None, _img, func_label2color=VOClabel2colormap, threshold=None, norm=False)
                IS_CAM = generate_vis(IS_cam_, None, _img,
                                                    func_label2color=VOClabel2colormap, threshold=None,
                                                    norm=False)

                writer.add_images('CAM', CAM, _b_idx + (batch_idx*_b_idx))
                writer.add_images('IS_CAM', IS_CAM, _b_idx + (batch_idx*_b_idx) )

                fg_conf_cam = IS_cam_[1:] # this is to test the IS performance
                # fg_conf_cam = _norm_cam_[1:]

                fg_conf_cam = np.pad(fg_conf_cam, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.50)
                fg_conf_cam = np.argmax(fg_conf_cam, axis=0)
                pred = crf_inference_label(_img, fg_conf_cam, n_labels=4)
                # print(pack['name'][_b_idx])
                # plt.imshow(pred)
                # plt.show()

                # temporary
                _gt_mask[_gt_mask>0] = 1
                pred[pred>0] = 1
                iou = compute_iou_segmentation(_gt_mask, pred)
                ious[str(int(_cat))].append(iou)
                names[str(int(_cat))].append(name)

                # print(ious)
                # print(names)
                #
                for key in ious:
                    _list = np.array(ious[key])
                    avg = np.mean(_list)
                    print(key, avg)
                print('-------------------------------------------------------------------------------------------------------')





