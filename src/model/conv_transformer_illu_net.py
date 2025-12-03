"""
Convolution-Transformer Illumination Network
A replacement for BilateralUpsampleNet that uses Conv-Transformer backbone for feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.arch.conv_transformer_backbone import ConvTransformerBackbone


class GuideNet(nn.Module):
    """Lightweight guide network for generating guidance maps"""
    
    def __init__(self, out_channel=1):
        super(GuideNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, out_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SliceNode(nn.Module):
    """Bilateral grid sampling with guide map"""
    
    def __init__(self):
        super(SliceNode, self).__init__()
    
    def forward(self, bilateral_grid, guidemap):
        """
        Args:
            bilateral_grid: Feature grid of shape (B, C, H_grid, W_grid, D)
            guidemap: Guide map of shape (B, 1, H, W)
        Returns:
            Sampled coefficients of shape (B, C, H, W)
        """
        device = bilateral_grid.device
        N, C, H_grid, W_grid, D_grid = bilateral_grid.shape
        B, _, H, W = guidemap.shape
        
        # Create coordinate grids
        hg, wg = torch.meshgrid(
            torch.arange(0, H, dtype=torch.float32, device=device),
            torch.arange(0, W, dtype=torch.float32, device=device),
            indexing='ij'
        )
        
        # Normalize coordinates to [-1, 1]
        hg = hg.unsqueeze(0).repeat(B, 1, 1) / (H - 1) * 2 - 1
        wg = wg.unsqueeze(0).repeat(B, 1, 1) / (W - 1) * 2 - 1
        
        # Normalize guidemap to [-1, 1]
        guidemap_norm = guidemap * 2 - 1
        
        # Concatenate to create sampling coordinates
        # Stack: [w, h, guide] for grid_sample
        coords = torch.stack([wg, hg, guidemap_norm.squeeze(1)], dim=-1)  # (B, H, W, 3)
        
        # Reshape bilateral grid for grid_sample: (B, C, H_grid, W_grid*D_grid)
        bilateral_grid = bilateral_grid.permute(0, 1, 2, 3, 4)  # (B, C, H_grid, W_grid, D_grid)
        B, C, H_g, W_g, D_g = bilateral_grid.shape
        bilateral_grid = bilateral_grid.reshape(B, C, H_g, W_g * D_g)
        
        # Perform grid sampling (simplified - assumes 2D grid sampling)
        # For 3D grid, we use 2D sampling on reshaped grid
        coords_2d = coords[..., :2].unsqueeze(2)  # (B, H, W, 1, 2) for grid_sample
        
        # Use bilinear interpolation
        coeff = F.grid_sample(
            bilateral_grid,
            coords_2d.squeeze(3),
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        )  # (B, C, H, W)
        
        return coeff


class ApplyCoeffs(nn.Module):
    """Apply affine transformation coefficients"""
    
    def __init__(self, coeff_dim=12):
        super(ApplyCoeffs, self).__init__()
        self.coeff_dim = coeff_dim
    
    def forward(self, coeff, full_res_input):
        """
        Apply affine transformation using coefficients
        Args:
            coeff: Coefficients of shape (B, coeff_dim, H, W)
            full_res_input: Input image of shape (B, 3, H, W)
        Returns:
            Transformed image of shape (B, 3, H, W)
        """
        # Affine transformation: output = A * input + b
        # For each channel: R = a11*r + a12*g + a13*b + a14
        
        if self.coeff_dim >= 12:
            # 3x4 affine matrix per pixel
            R = (
                torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True)
                + coeff[:, 3:4, :, :]
            )
            G = (
                torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True)
                + coeff[:, 7:8, :, :]
            )
            B = (
                torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True)
                + coeff[:, 11:12, :, :]
            )
            return torch.cat([R, G, B], dim=1)
        else:
            # Simpler transformation if fewer coefficients
            return full_res_input * coeff


class ConvTransformerIlluNet(nn.Module):
    """
    Illumination Network using Convolution-Transformer backbone
    Replaces BilateralUpsampleNet with better global context understanding
    """
    
    def __init__(self, opt=None, coeff_dim=12, num_blocks=4, base_channels=32):
        super(ConvTransformerIlluNet, self).__init__()
        
        self.opt = opt
        self.coeff_dim = coeff_dim
        self.guide_features = []
        self.guidemap = None
        self.illu_map = None
        self.slice_coeffs = None
        
        # Guide network for generating guide maps
        self.guide_net = GuideNet(out_channel=1)
        
        # Main backbone: Conv-Transformer for coefficient generation
        self.backbone = ConvTransformerBackbone(
            in_channels=3,
            out_channels=coeff_dim,
            num_blocks=num_blocks,
            base_channels=base_channels,
            num_heads=8,
            mlp_ratio=4.0
        )
        
        # Bilateral grid and slicing
        self.slice = SliceNode()
        self.apply_coeffs = ApplyCoeffs(coeff_dim=coeff_dim)
    
    def forward(self, lowres, fullres):
        """
        Args:
            lowres: Low-resolution input (B, 3, 256, 256)
            fullres: Full-resolution input (B, 3, H, W)
        Returns:
            Enhanced output image (B, 3, H, W)
        """
        # Generate coefficients from low-res input using Conv-Transformer backbone
        coefficients = self.backbone(lowres)  # (B, coeff_dim, 256, 256)
        
        # Store guide features
        try:
            self.guide_features = self.backbone.guide_features
        except Exception:
            pass
        
        # Generate guide map from full-res input
        guide = self.guide_net(fullres)  # (B, 1, H, W)
        self.guidemap = guide
        
        # Upsample coefficients to full resolution
        if coefficients.shape[-2:] != fullres.shape[-2:]:
            coefficients = F.interpolate(
                coefficients,
                size=fullres.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Slice coefficients using guide map
        self.slice_coeffs = coefficients
        
        # Apply coefficients to full-resolution input
        out = self.apply_coeffs(coefficients, fullres)
        
        return out
