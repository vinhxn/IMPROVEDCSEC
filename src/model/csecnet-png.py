# Design a generator that reconstructs pseudo-normal feature maps 
# from over-exposed or under-exposed inputs.
import os.path as osp
import pdb

import kornia as kn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

import utils.util as util
from globalenv import *

from .arch.cross_nonlocal import CrossNonLocalBlock
from .arch.nonlocal_block_embedded_gaussian import NONLocalBlock2D
from .basic_loss import *
from .dconv import ColorDeformConv2d
from .single_net_basemodel import SingleNetBaseModel
from .pseudo_normal_generator import PseudoNormalGenerator

class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss


class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(opt, DeepWBNet(opt[RUNTIME]), [TRAIN, VALID])

        self.pixel_loss = tanh_L1Loss()
        self.weighted_loss = WeightedL1Loss()
        self.tvloss = L_TV()
        self.ltv2 = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        self.histloss = HistogramLoss()
        self.vggloss = VGGLoss(model="vgg16", shift=2)
        self.vggloss.train()
        self.inter_histloss = IntermediateHistogramLoss()
        self.sparse_weight_loss = SparseWeightLoss()

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, output_batch = super().training_step_forward(
            batch, batch_idx
        )
        # Pseudo-Normal Ground Truth
        pseudo_normal_gt = batch["pseudo_normal_gt"]

        # Generator Loss
        pseudo_normal_pred = self.net.pseudo_normal_generator(input_batch)
        # pseudo_normal_loss = F.mse_loss(pseudo_normal_pred, pseudo_normal_gt)
        # Resize pseudo_normal_gt to match pseudo_normal_pred
        pseudo_normal_gt_resized = F.interpolate(pseudo_normal_gt, size=pseudo_normal_pred.shape[2:], mode='bilinear', align_corners=False)
        pseudo_normal_loss = F.mse_loss(pseudo_normal_pred, pseudo_normal_gt_resized)

        # Existing Losses
        loss_lambda_map = {
            L1_LOSS: lambda: self.pixel_loss(output_batch, gt_batch),
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean())
            * 0.5,
            COS_LOSS
            + "2": lambda: 1
            - F.sigmoid(self.cos(output_batch, gt_batch).mean()),
            LTV_LOSS: lambda: self.tvloss(output_batch),
            "tvloss1": lambda: self.tvloss(self.net.res[ILLU_MAP])
            + self.tvloss(self.net.res[BRIGHTEN_INPUT]),
            "tvloss2": lambda: self.tvloss(self.net.res[INVERSE_ILLU_MAP])
            + self.tvloss(self.net.res[DARKEN_INPUT]),
            "tvloss1_new": lambda: self.ltv2(
                input_batch, self.net.res[ILLU_MAP], 1
            ),
            "tvloss2_new": lambda: self.ltv2(
                1 - input_batch, self.net.res[INVERSE_ILLU_MAP], 1
            ),
            "illumap_loss": lambda: F.mse_loss(
                self.net.res[ILLU_MAP], 1 - self.net.res[INVERSE_ILLU_MAP]
            ),
            WEIGHTED_LOSS: lambda: self.weighted_loss(
                input_batch.detach(), output_batch, gt_batch
            ),
            SSIM_LOSS: lambda: kn.losses.ssim_loss(
                output_batch, gt_batch, window_size=11
            ),
            PSNR_LOSS: lambda: kn.losses.psnr_loss(
                output_batch, gt_batch, max_val=1.0
            ),
            HIST_LOSS: lambda: self.histloss(output_batch, gt_batch),
            INTER_HIST_LOSS: lambda: self.inter_histloss(
                input_batch,
                gt_batch,
                self.net.res[BRIGHTEN_INPUT],
                self.net.res[DARKEN_INPUT],
            ),
            VGG_LOSS: lambda: self.vggloss(input_batch, gt_batch),
        }
        if self.opt[RUNTIME][DEFORM]:
            loss_lambda_map.update(
                {
                    NORMAL_EX_LOSS: lambda: self.pixel_loss(
                        self.net.res[NORMAL], gt_batch
                    )
                }
            )
        
        loss = self.calc_and_log_losses(loss_lambda_map) + pseudo_normal_loss
                
        # logging images:
        self.log_training_iogt_img(batch)
        return loss

    def validation_step(self, batch, batch_idx):...
        
    def test_step(self, batch, batch_ix):
        super().test_step(batch, batch_ix)

        # save intermediate results
        for k, v in self.net.res.items():
            dirpath = Path(self.opt[IMG_DIRPATH]) / k
            fname = osp.basename(batch[INPUT_FPATH][0])
            if "illu" in k:
                util.mkdir(dirpath)
                torchvision.utils.save_image(v[0].unsqueeze(1), dirpath / fname)
            elif k == "guide_features":
                util.mkdir(dirpath)
                max_size = v[-1][-1].shape[-2:]
                final = []
                for level_guide in v:
                    gs = [F.interpolate(g, max_size) for g in level_guide]
                    final.extend(gs)
                region_num = final[0].shape[1]
                final = torch.stack(final).argmax(axis=2).float() / region_num
                torchvision.utils.save_image(final, dirpath / fname)
            else:
                self.save_img_batch(v, dirpath, fname)


class DeepWBNet(nn.Module):
    def build_illu_net(self):
        from .bilateralupsamplenet import BilateralUpsampleNet

        return BilateralUpsampleNet(self.opt[BUNET])

    def backbone_forward(self, net, x):
        low_x = self.down_sampler(x)
        res = net(low_x, x)
        self.res.update({"guide_features": net.guide_features})
        return res

    def __init__(self, opt=None):
        super(DeepWBNet, self).__init__()
        self.opt = opt
        self.res = {}
        self.down_sampler = lambda x: F.interpolate(
            x, size=(256, 256), mode="bicubic", align_corners=False
        )
        self.illu_net = self.build_illu_net()

        # Initialize the Generator
        self.pseudo_normal_generator = PseudoNormalGenerator()

        # Use non-local block for efficiency.
        nf = 32
        self.out_net = nn.Sequential(
            nn.Conv2d(9, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(nf, sub_sample="bilinear", bn_layer=False),
            nn.Conv2d(nf, nf, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 3, 1),
            NONLocalBlock2D(3, sub_sample="bilinear", bn_layer=False),
        )
        # If you have enough computing resources, you can also use the following code to replace the above code.
        # Remember to modify the corresponding code in the forward function.
        # self.out_net = CrossNonLocalBlock(in_channels=3, inter_channels=16)

        if opt[DEFORM]:
            self.over_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )
            self.under_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )

    def decomp(self, x1, illu_map):
        return x1 / (torch.where(illu_map < x1, x1, illu_map.float()) + 1e-7)

    def forward(self, x):
        x1 = x
        inverse_x1 = 1 - x1

        # Backbone
        illu_map = self.backbone_forward(self.illu_net, x1)
        inverse_illu_map = self.backbone_forward(self.illu_net, inverse_x1)

        # Enhancement
        illu_map = util.rgb2gray(illu_map)
        inverse_illu_map = util.rgb2gray(inverse_illu_map)
        brighten_x1 = self.decomp(x1, illu_map)
        inverse_x2 = self.decomp(inverse_x1, inverse_illu_map)
        darken_x1 = 1 - inverse_x2

        self.res.update({
            INVERSE: 1 - x,
            ILLU_MAP: illu_map,
            INVERSE_ILLU_MAP: inverse_illu_map,
            BRIGHTEN_INPUT: brighten_x1,
            DARKEN_INPUT: darken_x1,
        })

        # Pseudo-Normal Generation: Replace previous logic with the generator
        # Resize each tensor to match the dimensions before concatenation
        brighten_x1 = F.interpolate(brighten_x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        darken_x1 = F.interpolate(darken_x1, size=x.shape[2:], mode='bilinear', align_corners=False)

        input_tensor = torch.cat([x, brighten_x1, darken_x1], dim=1)
        input_tensor_resized = F.interpolate(input_tensor, size=(512, 512), mode='bilinear', align_corners=False)
        pseudo_normal = self.pseudo_normal_generator(input_tensor_resized)
        
        self.res.update({NORMAL: pseudo_normal})

        # Deformation
        brighten_x1_resized = F.interpolate(brighten_x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        darken_x1_resized = F.interpolate(darken_x1, size=x.shape[2:], mode='bilinear', align_corners=False)
        brighten_x2 = self.over_deform(x=pseudo_normal, ref=brighten_x1_resized)
        darken_x2 = self.under_deform(x=pseudo_normal, ref=darken_x1_resized)

        # Modulation
        out = self.out_net(
            torch.cat([pseudo_normal, brighten_x2, darken_x2], dim=1)
        )
        w1 = torch.sigmoid(out[:, 0, ...].unsqueeze(1))
        w2 = torch.sigmoid(out[:, 1, ...].unsqueeze(1))
        w3 = torch.sigmoid(out[:, 2, ...].unsqueeze(1))
        #out = x * w1 + brighten_x1 * w2 + darken_x1 * w3

        w1_resized = F.interpolate(w1, size=x.shape[2:], mode='bilinear', align_corners=False)
        w2_resized = F.interpolate(w2, size=x.shape[2:], mode='bilinear', align_corners=False)
        w3_resized = F.interpolate(w3, size=x.shape[2:], mode='bilinear', align_corners=False)  
        # Perform the element-wise operations with resized tensors
        out = x * w1_resized + brighten_x1_resized * w2_resized + darken_x1_resized * w3_resized

        self.res.update({
            BRIGHTEN_OFFSET: brighten_x2,
            DARKEN_OFFSET: darken_x2,
        })

        assert out.shape == x.shape
        return out

