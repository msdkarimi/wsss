import torch.nn as nn

from net import resnet50
import torch
import torch.nn.functional as F
from utils import CPFE, ChannelwiseAttention, normalize_tensor
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, num_cls):
        super(Net, self).__init__()
        self.resnet50 = resnet50(pretrained=True, strides=(2, 2, 2, 1), dilations=(1, 1, 1, 1))
        self.num_cls = num_cls

        self.stage0 = nn.Sequential(self.resnet50.conv1, self.resnet50.bn1, self.resnet50.relu, self.resnet50.maxpool)
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        self.side1 = nn.Conv2d(256, 128, 1, bias=False)
        self.side2 = nn.Conv2d(512, 128, 1, bias=False)
        self.side3 = nn.Conv2d(1024, 256, 1, bias=False)
        self.side4 = nn.Conv2d(2048, 256, 1, bias=False)

        self.cpef_1 = CPFE(layer_name='1')
        self.cpef_2 = CPFE(layer_name='2')
        self.cpef_3 = CPFE(layer_name='3')
        self.cpef_4 = CPFE(layer_name='4')
        self.cha_att = ChannelwiseAttention(in_channels=512)


        self.classifier = nn.Conv2d(2048, self.num_cls - 1, 1, bias=False)

        self.backbone = nn.ModuleList([self.stage0, self.stage1, self.stage2, self.stage3, self.stage4])
        # self.newly_added = nn.ModuleList([self.classifier, self.side1, self.side2, self.side3, self.side4])
        self.newly_added = nn.ModuleList([self.classifier, self.cpef_1, self.cpef_2, self.cpef_3, self.cpef_4, self.cha_att])

    def forward(self, x, valid_mask, name=None, category=None):
        N, C, H, W = x.size()

        # forward
        x0 = self.stage0(x)
        x1 = self.stage1(x0)
        # x2 = self.stage2(x1).detach()
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        # side1 = self.side1(x1.detach())
        # side2 = self.side2(x2.detach())
        # side3 = self.side3(x3.detach())
        # side4 = self.side4(x4.detach())

        cpef_1 = self.cpef_1(x1.detach())
        cpef_2 = self.cpef_2(x2.detach())
        cpef_3 = self.cpef_3(x3.detach())
        cpef_4 = self.cpef_4(x4.detach())

        # hie_fea = torch.cat(
        #     [F.interpolate(side1 / (torch.norm(side1, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
        #      F.interpolate(side2 / (torch.norm(side2, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
        #      F.interpolate(side3 / (torch.norm(side3, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear'),
        #      F.interpolate(side4 / (torch.norm(side4, dim=1, keepdim=True) + 1e-5), side3.shape[2:], mode='bilinear')],
        #     dim=1)

        cpef_1 = normalize_tensor(cpef_1, interpolate_res=cpef_4.shape[2:])
        cpef_2 = normalize_tensor(cpef_2, interpolate_res=cpef_4.shape[2:])
        cpef_3 = normalize_tensor(cpef_3)
        cpef_4 = normalize_tensor(cpef_4)

        hie_fea_2 = torch.cat([cpef_1, cpef_2, cpef_3, cpef_4], dim=1)
        hie_fea_ca, ca_act_reg = self.cha_att(hie_fea_2)
        # layer1234_cpfe_feats = torch.mul(hie_fea_ca, hie_fea_2) # this helps for saliency detection
        layer1234_cpfe_feats = normalize_tensor(torch.mul(hie_fea_ca, hie_fea_2)) # this helps for saliency detection


        sem_feature = x4
        cam = self.classifier(x4)
        score = F.adaptive_avg_pool2d(cam, 1)

        # initialize background map
        norm_cam = F.relu(cam)
        norm_cam = norm_cam / (F.adaptive_max_pool2d(norm_cam, (1, 1)) + 1e-5)
        cam_bkg = 1 - torch.max(norm_cam, dim=1)[0].unsqueeze(1)
        norm_cam = torch.cat([cam_bkg, norm_cam], dim=1)
        norm_cam = F.interpolate(norm_cam, cpef_3.shape[2:], mode='bilinear', align_corners=True) * valid_mask

        _mask = torch.sum(norm_cam.detach().clone(), dim=1)
        _mask = (_mask > 0.5).cpu().numpy() * 1
        fut_for_mask = torch.sum(norm_cam.detach(), dim=1).cpu().numpy()
        activations = []
        for idx_image in range(norm_cam.shape[0]):
            _init_seed = self.get_seeds_from_cam(_mask[idx_image], fut_for_mask[idx_image])
            num_sp = max(len(torch.nonzero(_init_seed[:, 0])), len(torch.nonzero(_init_seed[:, 1])))
            # TODO handle the case of zero
            if (num_sp == 0) or (num_sp == 1):
                _prototype = self.Weighted_GAP(layer1234_cpfe_feats[idx_image], _mask[idx_image])  # 1 x c x 1 x 1
                _activation = self.activate_features(layer1234_cpfe_feats[idx_image], _prototype)
                activations.append(_activation * valid_mask[idx_image])
                continue

            s_seed_ = _init_seed[:num_sp, :].cuda()

            _prototypes = self.update_super_pixel_centroid(layer1234_cpfe_feats[idx_image], _mask[idx_image], s_seed_, 10)
            _activation = self.activate_features(layer1234_cpfe_feats[idx_image], _prototypes)
            activations.append(_activation * valid_mask[idx_image])

        IS_cam = torch.stack(activations, dim=0)




        # seeds, probs = self.get_seed(norm_cam.clone(), valid_mask.clone(), sem_feature.clone())
        # prototypes = self.get_prototype(seeds, layer1234_cpfe_feats)
        # # prototypes = self.get_prototype(seeds, hie_fea)
        # IS_cam = self.reactivate(prototypes, layer1234_cpfe_feats)
        # # IS_cam = self.reactivate(prototypes, hie_fea)

        # return {"score": score, "cam": norm_cam, "seeds": seeds, "prototypes": prototypes, "IS_cam": IS_cam,
        #         "probs": probs}

        return {"score": score, "cam": norm_cam, "IS_cam": IS_cam}

    @staticmethod
    def Weighted_GAP(supp_feat, mask):
        mask = torch.from_numpy(mask).cuda().unsqueeze(0).float()
        supp_feat = supp_feat * mask
        feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
        area = F.avg_pool2d(mask, (supp_feat.size()[1], supp_feat.size()[2])) * feat_h * feat_w + 0.0005
        supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
        return supp_feat.squeeze(-1)


    def get_seeds_from_cam(self, _mask, cam, max_num_sp=5, avg_sp_area=3):
        segments_x = np.zeros(max_num_sp, dtype=np.int64)
        segments_y = np.zeros(max_num_sp, dtype=np.int64)

        nz = np.nonzero(_mask)

        if len(nz[0]) != 0:
            p = [np.min(nz[0]), np.min(nz[1])]
            pend = [np.max(nz[0]), np.max(nz[1])]

            # Cropping to bounding box around ROI
            m_np_roi = np.copy(_mask)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]
            cam_roi = np.copy(cam)[p[0]:pend[0] + 1, p[1]:pend[1] + 1]

            # Normalize CAM values
            cam_roi = (cam_roi - cam_roi.min()) / (cam_roi.max() - cam_roi.min() + 1e-6)

            # Adaptive number of seeds
            mask_area = (m_np_roi == 1).sum()
            num_sp = int(min((np.array(mask_area) / avg_sp_area).round(), max_num_sp))
        else:
            num_sp = 0

        if (num_sp != 0) and (num_sp != 1):
            for i in range(num_sp):
                # Weighted Distance Transform (biasing towards high-CAM areas)
                weighted_dtrans = distance_transform_edt(m_np_roi * cam_roi)
                weighted_dtrans = gaussian_filter(weighted_dtrans, sigma=0.1)

                coords1 = np.nonzero(weighted_dtrans == np.max(weighted_dtrans))
                segments_x[i] = coords1[0][0]
                segments_y[i] = coords1[1][0]

                # Remove the selected seed point from the region
                m_np_roi[segments_x[i], segments_y[i]] = False
                segments_x[i] += p[0]
                segments_y[i] += p[1]

        segments = np.concatenate([segments_x[..., np.newaxis], segments_y[..., np.newaxis]], axis=1)  # max_num_sp x 2
        segments = torch.from_numpy(segments)

        return segments


    def get_seed(self, norm_cam, label, feature):

        n, c, h, w = norm_cam.shape

        # iou evalution
        seeds = torch.zeros((n, h, w, c)).cuda()
        feature_s = feature.view(n, -1, h * w)
        feature_s = feature_s / (torch.norm(feature_s, dim=1, keepdim=True) + 1e-5)
        correlation = F.relu(torch.matmul(feature_s.transpose(2, 1), feature_s), inplace=True).unsqueeze(
            1)  # [n,1,h*w,h*w]
        # correlation = correlation/torch.max(correlation, dim=-1)[0].unsqueeze(-1) #[n,1,h*w,h*w]
        cam_flatten = norm_cam.view(n, -1, h * w).unsqueeze(2)  # [n,21,1,h*w]
        inter = (correlation * cam_flatten).sum(-1)
        union = correlation.sum(-1) + cam_flatten.sum(-1) - inter
        miou = (inter / union).view(n, self.num_cls, h, w)  # [n,21,h,w]
        miou[:, 0] = miou[:, 0] * 0.5
        probs = F.softmax(miou, dim=1)
        belonging = miou.argmax(1)
        seeds = seeds.scatter_(-1, belonging.view(n, h, w, 1), 1).permute(0, 3, 1, 2).contiguous()

        seeds = seeds * label
        return seeds, probs


    @staticmethod
    def update_super_pixel_centroid(feat, mask, sp_init_center, n_iter):

        _sp_init_center = feat[:, sp_init_center[:, 0], sp_init_center[:, 1]]  # c x num_sp (sp_seed)
        sp_init_center = torch.cat([_sp_init_center, sp_init_center.transpose(1, 0).float()], dim=0)  # (c + xy) x num_sp


        c_xy, num_sp = sp_init_center.size()
        _, h, w = feat.size()
        h_coords = torch.arange(h).view(h, 1).contiguous().repeat(1, w).unsqueeze(0).float().cuda()
        w_coords = torch.arange(w).repeat(h, 1).unsqueeze(0).float().cuda()
        supp_feat = torch.cat([feat, h_coords, w_coords], 0)
        supp_feat_roi = supp_feat[:, (mask == 1).squeeze()]  # (C + xy) x num_roi


        # # could be googd point to add cuntrustive loss
        num_roi = supp_feat_roi.size(1)
        supp_feat_roi_rep = supp_feat_roi.unsqueeze(-1).repeat(1, 1, num_sp)
        sp_center = torch.zeros_like(sp_init_center).cuda()  # (C + xy) x num_sp

        for i in range(n_iter):
            # Compute association between each pixel in RoI and superpixel
            if i == 0:
                sp_center_rep = sp_init_center.unsqueeze(1).repeat(1, num_roi, 1)
            else:
                sp_center_rep = sp_center.unsqueeze(1).repeat(1, num_roi, 1)
            assert supp_feat_roi_rep.shape == sp_center_rep.shape  # (C + xy) x num_roi x num_sp
            # here!!!!!!
            dist = torch.pow(supp_feat_roi_rep - sp_center_rep, 2.0) + 1e-8
            if torch.isnan(dist).any():
                print("in model!!! 154")
            feat_dist = dist[:-2, :, :].sum(0)
            spat_dist = dist[-2:, :, :].sum(0)
            total_dist = torch.pow(feat_dist + spat_dist / 3, 0.5)
            p2sp_assoc = torch.neg(total_dist).exp()  # make sure for the distance value
            p2sp_assoc = p2sp_assoc / (p2sp_assoc.sum(0, keepdim=True))  # num_roi x num_sp

            sp_center = supp_feat_roi_rep * p2sp_assoc.unsqueeze(0)  # (C + xy) x num_roi x num_sp , Multiplying the features of the image by the association matrix essentially filters out irrelevant features and focuses the learning on the most relevant regions
            sp_center = sp_center.sum(1)
        result = sp_center[:-2, :]

        return result



    def activate_features(self, hie_fea, _prototypes):
        # is_cam_aggregation = torch.zeros (hie_fea.shape[1:]).cuda()
        _list_is_cam_aggregation = []
        for _prototype_idx in range(int(_prototypes.shape[-1])):
            _pro = _prototypes[:, _prototype_idx]
            # activation = F.relu(torch.cosine_similarity(prototype_features, _pro.unsqueeze(-1).unsqueeze(-1), dim=0, eps=1e-8))
            tensor_1_normalized = _pro / (torch.norm(_pro, dim=0, keepdim=True) + 1e-8)
            tensor_2_normalized = hie_fea / (torch.norm(hie_fea, dim=0, keepdim=True) + 1e-8)

            # Compute cosine similarity
            activation = torch.sum(tensor_1_normalized.unsqueeze(-1).unsqueeze(-1) * tensor_2_normalized, dim=0)
            # is_cam_aggregation = is_cam_aggregation + activation
            _list_is_cam_aggregation.append(activation)
        # is_cam_aggregation = is_cam_aggregation / _prototypes.shape[-1]
        # is_cam_aggregation = torch.stack(_list_is_cam_aggregation).mean(dim=0) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        is_cam_aggregation = torch.stack(_list_is_cam_aggregation).mean(dim=0) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # total_is_cams.append(is_cam_aggregation.unsqueeze(0))


        return F.relu(is_cam_aggregation.unsqueeze(0))



    def get_prototype(self, seeds, feature):
        n, c, h, w = feature.shape
        seeds = F.interpolate(seeds, feature.shape[2:], mode='nearest')
        crop_feature = seeds.unsqueeze(2) * feature.unsqueeze(
            1)  # seed:[n,21,1,h,w], feature:[n,1,c,h,w], crop_feature:[n,21,c,h,w]
        prototype = F.adaptive_avg_pool2d(crop_feature.view(-1, c, h, w), (1, 1)).view(n, self.num_cls, c, 1,
                                                                                       1)  # prototypes:[n,21,c,1,1]
        return prototype

    def reactivate(self, prototype, feature):
        IS_cam = F.relu(torch.cosine_similarity(feature.unsqueeze(1), prototype,
                                                dim=2))  # feature:[n,1,c,h,w], prototypes:[n,21,c,1,1], crop_feature:[n,21,h,w]
        IS_cam = F.interpolate(IS_cam, feature.shape[2:], mode='bilinear', align_corners=True)
        return IS_cam


    # def train(self, mode=True):
    #     for p in self.resnet50.conv1.parameters():
    #         p.requires_grad = False
    #     for p in self.resnet50.bn1.parameters():
    #         p.requires_grad = False

    def trainable_parameters(self):
        return (list(self.backbone.parameters()), list(self.newly_added.parameters()))
