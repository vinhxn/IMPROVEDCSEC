"""
Convolution-Transformer Backbone for Feature Extraction
Combines convolutional layers with transformer attention for capturing both local and global context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with multi-head self-attention and feed-forward network"""
    
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C) for transformer
        x_flat = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # Self-attention with residual connection
        x_norm = self.norm1(x_flat)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x_flat = x_flat + attn_out
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x_flat)
        ffn_out = self.mlp(x_norm)
        x_flat = x_flat + ffn_out
        
        # Reshape back to (B, C, H, W)
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x


class ConvBlock(nn.Module):
    """Convolutional block with batch norm and activation"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTransformerBlock(nn.Module):
    """Combined convolution and transformer block"""
    
    def __init__(self, channels, num_heads=8, mlp_ratio=4.0):
        super(ConvTransformerBlock, self).__init__()
        
        # Convolutional pathway for local feature extraction
        self.conv_path = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        
        # Transformer pathway for global context
        self.transformer_block = TransformerEncoderBlock(
            dim=channels,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Parallel pathways
        conv_out = self.conv_path(x)
        trans_out = self.transformer_block(x)
        
        # Concatenate and fuse
        fused = torch.cat([conv_out, trans_out], dim=1)
        out = self.fusion(fused)
        
        # Residual connection
        out = out + x
        
        return out


class ConvTransformerBackbone(nn.Module):
    """
    Backbone combining convolutional and transformer layers for feature extraction
    Suitable as a replacement for HistUNet in illumination network
    """
    
    def __init__(self, in_channels=3, out_channels=12, num_blocks=4, 
                 base_channels=32, num_heads=8, mlp_ratio=4.0):
        super(ConvTransformerBackbone, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initial convolution
        self.input_conv = ConvBlock(in_channels, base_channels, kernel_size=7, stride=1, padding=3)
        
        # Encoder with Conv-Transformer blocks
        self.encoder_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        in_ch = base_channels
        for i in range(num_blocks):
            # Conv-Transformer block
            self.encoder_blocks.append(
                ConvTransformerBlock(in_ch, num_heads=num_heads, mlp_ratio=mlp_ratio)
            )
            
            # Downsample (skip last)
            if i < num_blocks - 1:
                out_ch = in_ch * 2
                self.downsample.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    )
                )
                in_ch = out_ch
        
        # Decoder with upsampling
        self.decoder_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        decoder_channels = [in_ch // (2 ** i) for i in range(num_blocks)]
        
        for i in range(num_blocks - 1):
            # Upsample
            self.upsample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(decoder_channels[i], decoder_channels[i+1],
                                     kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(decoder_channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
            
            # Conv-Transformer block
            self.decoder_blocks.append(
                ConvTransformerBlock(decoder_channels[i+1], num_heads=num_heads, mlp_ratio=mlp_ratio)
            )
        
        # Output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1, stride=1, padding=0)
        )
        
        self.guide_features = []
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        Returns:
            Output features of shape (B, out_channels, H, W)
        """
        # Input processing
        x = self.input_conv(x)
        
        # Encoder
        encoder_outputs = [x]
        for i, (block, down) in enumerate(zip(self.encoder_blocks[:-1], self.downsample)):
            x = block(x)
            encoder_outputs.append(x)
            x = down(x)
        
        # Bottleneck
        x = self.encoder_blocks[-1](x)
        
        # Decoder with skip connections
        for i, (up, block) in enumerate(zip(self.upsample, self.decoder_blocks)):
            x = up(x)
            # Skip connection
            x = x + encoder_outputs[-(i+2)]
            x = block(x)
        
        # Output
        x = self.output_conv(x)
        
        return x
