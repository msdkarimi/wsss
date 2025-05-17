from sklearn.utils.class_weight import compute_sample_weight
import torch.nn.functional as F
# from model import Net
from model_cpfe_multi import Net
from dataset_loader_wsss_multi_label import SupConDataset
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
from utils import PolyOptimizerSGD, AverageMeter, convert_image, generate_vis, VOClabel2colormap
from torch.utils.tensorboard import SummaryWriter
import numpy as np


if __name__ == '__main__':
    model = Net(num_cls=13).cuda()
    model.train()
    param_groups = model.trainable_parameters()
    lr = 9e-3
    wt_dec = 1e-4
    epochs = 200
    print_freq = 10

    writer = SummaryWriter('logs_multi/')

    # train_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\copied_focal\copied_focal\dataset',
    train_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction',
                                  mode='train',
                                  transform=transforms.Compose(
                                      [
                                          transforms.Resize((224, 224)),
                                          transforms.RandomGrayscale(),
                                          transforms.ToTensor(),
                                      ]))
    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

    val_dataset = SupConDataset(root_dir='C:\\Users\massoud\PycharmProjects\\700perizie_cleaning\wsss_image_annotation_extaction',
                                  mode='val',
                                  transform=transforms.Compose(
                                      [
                                          transforms.Resize((224, 224)),
                                          transforms.ToTensor(),
                                      ]))
    val_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False)

    max_step = epochs*len(train_loader)

    optimizer = PolyOptimizerSGD([
        {'params': param_groups[0], 'lr': lr, 'weight_decay': wt_dec},
        # {'params': param_groups[1], 'lr': 10 * lr, 'weight_decay': wt_dec}
        {'params': param_groups[1], 'lr': lr * 10 , 'weight_decay': wt_dec}
    ], lr=lr, weight_decay=wt_dec, max_step=max_step)

    avg_meter = AverageMeter()


    for epoch in range(epochs):
        for batch_idx, pack in enumerate(train_loader):
            img = pack['images'].cuda()
            n, c, h, w = img.shape
            label = pack['labels'].cuda(non_blocking=True).unsqueeze(-1).unsqueeze(-1)

            valid_mask = pack['valid_mask'].cuda()
            valid_mask[:,1:] = valid_mask[:,1:] * label
            valid_mask_lowres = F.interpolate(valid_mask, size=(h // 16, w // 16), mode='nearest')
            outputs = model.forward(img, valid_mask_lowres, name=pack['name'], category=pack['cat'])


            score = outputs['score']
            norm_cam = outputs['cam']
            IS_cam = outputs['IS_cam']

            lossCLS = F.multilabel_soft_margin_loss(score, label, weight=torch.from_numpy(np.array(
                [0.36688435, 0.2992512,  2.0253396,  0.27134957,
                 1.2743261,  1.04375337,1.22829531, 4.22276688,
                 3.10616987, 5.00839793, 2.11830601, 3.19843234])).cuda())

            IS_cam = IS_cam / (F.adaptive_max_pool2d(IS_cam, (1, 1)) + 1e-5)
            lossGSC = torch.mean(torch.abs(norm_cam - IS_cam)) * 2

            losses = lossCLS + lossGSC
            avg_meter.add({'lossCLS': lossCLS.item(), 'lossGSC': lossGSC.item()})

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            if (optimizer.global_step - 1) % print_freq == 0:
                print('step:%5d/%5d' % (optimizer.global_step - 1, max_step),
                      'lossCLS:%.4f' % (avg_meter.pop('lossCLS')),
                      'lossGSC:%.4f' % (avg_meter.pop('lossGSC')),
                      'lr: %.4f' % (optimizer.param_groups[0]['lr']), flush=True)

            if (optimizer.global_step - 1) % print_freq:
                _b_idx = 0
                _img = convert_image(img[_b_idx])
                norm_cam = F.interpolate(norm_cam, _img.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                IS_cam = F.interpolate(IS_cam, _img.shape[1:], mode='bilinear')[0].detach().cpu().numpy()
                CAM = generate_vis(norm_cam, None, _img, func_label2color=VOClabel2colormap, threshold=None, norm=False)
                IS_CAM = generate_vis(IS_cam, None, _img,
                                                    func_label2color=VOClabel2colormap, threshold=None,
                                                    norm=False)

                writer.add_images('CAM', CAM, optimizer.global_step)
                writer.add_images('IS_CAM', IS_CAM, optimizer.global_step)

            if (optimizer.global_step - 1) % 1000 == 0:
                torch.save(model.state_dict(), f'./checkpoint_multi/model_{optimizer.global_step}.pth')


