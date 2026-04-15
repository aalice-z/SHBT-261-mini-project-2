# -*- coding: utf-8 -*-
"""
U-Net Implementation for Semantic Segmentation
Implements the U-Net architecture as described in:
"U-Net: Convolutional Networks for Biomedical Image Segmentation"
Ronneberger et al., MICCAI 2015
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class DoubleConv(nn.Module):
    """Two consecutive convolutional blocks with ReLU activation."""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downsampling block with max pooling followed by double convolution."""
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connections."""
    def __init__(self, in_channels, out_channels, use_conv_transpose=True):
        super(UpBlock, self).__init__()
        if use_conv_transpose:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        else:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        
        # Handle size mismatch due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.pad(x, (0, skip.shape[3] - x.shape[3], 
                          0, skip.shape[2] - x.shape[2]))
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """
    U-Net architecture for semantic segmentation.
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        num_classes (int): Number of output classes (default: 21 for PASCAL VOC)
        features (list): Number of features at each level (default: [64, 128, 256, 512])
        use_conv_transpose (bool): Use ConvTranspose2d for upsampling (default: True)
    """
    def __init__(self, in_channels=3, num_classes=21, features=None, use_conv_transpose=True):
        super(UNet, self).__init__()
        
        if features is None:
            features = [64, 128, 256, 512]
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.features = features

        # Encoder (Contraction Path)
        self.inc = DoubleConv(in_channels, features[0])
        self.down1 = DownBlock(features[0], features[1])
        self.down2 = DownBlock(features[1], features[2])
        self.down3 = DownBlock(features[2], features[3])
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(features[3], features[3] * 2)
        )
        
        # Decoder (Expansion Path)
        self.up3 = UpBlock(features[3] * 2, features[3], use_conv_transpose)
        self.up2 = UpBlock(features[3], features[2], use_conv_transpose)
        self.up1 = UpBlock(features[2], features[1], use_conv_transpose)
        self.up0 = UpBlock(features[1], features[0], use_conv_transpose)
        
        # Output layer
        self.outc = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        x5 = self.bottleneck(x4)
        
        # Decoder with skip connections
        x = self.up3(x5, x4)
        x = self.up2(x, x3)
        x = self.up1(x, x2)
        x = self.up0(x, x1)
        
        # Output
        x = self.outc(x)
        return x


class DeepLabV3Lite(nn.Module):
    """
    Lightweight DeepLabV3-like model using ResNet50 backbone.
    Suitable for VOC segmentation with dilated convolutions.
    """
    def __init__(self, num_classes=21, pretrained=True):
        super(DeepLabV3Lite, self).__init__()
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        
        # ASPP module (simplified)
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = DecoderModule(256, num_classes)
    
    def forward(self, x):
        size = x.shape[2:]
        
        # Backbone
        features = self.backbone(x)
        
        # ASPP
        features = self.aspp(features)
        
        # Decoder
        logits = self.decoder(features)
        
        # Upsample to original size
        logits = F.interpolate(logits, size=size, mode='bilinear', align_corners=False)
        
        return logits


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module."""
    def __init__(self, in_channels, out_channels, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()
        
        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Multiple atrous convolutions
        self.aspp_modules = nn.ModuleList([
            AtrousConv(in_channels, out_channels, rate)
            for rate in atrous_rates
        ])
        
        # Image pooling
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Project and merge
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        res = self.conv1x1(x)
        
        for aspp_module in self.aspp_modules:
            res_atrous = aspp_module(x)
            res = torch.cat([res, res_atrous], dim=1)
        
        # Image pooling
        size = x.shape[2:]
        pool = self.image_pool(x)
        pool = F.interpolate(pool, size=size, mode='bilinear', align_corners=False)
        res = torch.cat([res, pool], dim=1)
        
        res = self.project(res)
        return res


class AtrousConv(nn.Module):
    """Atrous (dilated) convolution module."""
    def __init__(self, in_channels, out_channels, rate):
        super(AtrousConv, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.atrous_conv(x)


class DecoderModule(nn.Module):
    """Decoder module for upsampling and feature refinement."""
    def __init__(self, in_channels, num_classes):
        super(DecoderModule, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.out = nn.Conv2d(128, num_classes, kernel_size=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)
        x = self.out(x)
        return x


def get_model(model_name='unet', num_classes=21, pretrained=False, **kwargs):
    """
    Factory function to instantiate segmentation models.
    
    Args:
        model_name (str): Name of model ('unet', 'deeplab')
        num_classes (int): Number of output classes
        pretrained (bool): Use pretrained weights
        **kwargs: Additional arguments for model
    
    Returns:
        Model instance
    """
    if model_name.lower() == 'unet':
        return UNet(num_classes=num_classes, **kwargs)
    elif model_name.lower() == 'deeplab':
        return DeepLabV3Lite(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model: {model_name}")


if __name__ == "__main__":
    # Test the models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test U-Net
    unet = UNet(in_channels=3, num_classes=21).to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    out = unet(x)
    print(f"U-Net input shape: {x.shape}")
    print(f"U-Net output shape: {out.shape}")
    print(f"U-Net parameters: {sum(p.numel() for p in unet.parameters()):,}")
    
    # Test DeepLab
    print("\n" + "="*50)
    deeplab = DeepLabV3Lite(num_classes=21, pretrained=False).to(device)
    out = deeplab(x)
    print(f"DeepLab input shape: {x.shape}")
    print(f"DeepLab output shape: {out.shape}")
    print(f"DeepLab parameters: {sum(p.numel() for p in deeplab.parameters()):,}")
