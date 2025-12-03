import torch
import torch.nn as nn

class PseudoNormalGenerator(nn.Module):
    def __init__(self):
        super(PseudoNormalGenerator, self).__init__()

        # Adjust the first layer to accept 9 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(9, 64, kernel_size=4, stride=2, padding=1),  # Input channels set to 9
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Bottleneck layers
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Upsampling and reconstruction
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # Output normalized to [-1, 1]
        )
    
    def forward(self, x):
        # Forward pass
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x