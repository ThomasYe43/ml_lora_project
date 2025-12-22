import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ChannelLoRAConv2d(nn.Module):
    """Feature-map LoRA: decompose (out×in) per spatial position into (out×r)(r×in)."""

    def __init__(self, conv: nn.Conv2d, r: int = 4, alpha: int = 8):
        super().__init__()
        self.conv = conv
        self.conv.requires_grad_(False)  # freeze base conv

        oc, ic, kh, kw = conv.weight.shape
        self.scale = alpha / float(r)
        self.r = r

        # A: (kh, kw, oc, r), B: (kh, kw, r, ic)
        self.lora_A = nn.Parameter(torch.randn(kh, kw, oc, r) / math.sqrt(r))
        self.lora_B = nn.Parameter(torch.zeros(kh, kw, r, ic))

    def forward(self, x):
        oc, ic, kh, kw = self.conv.weight.shape

        # delta[h, w, o, i] = sum_r A[h,w,o,r] * B[h,w,r,i]
        delta = torch.einsum('hwor,hwri->oihw', self.lora_A, self.lora_B)
        w = self.conv.weight + self.scale * delta

        return F.conv2d(
            x,
            w,
            bias=self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )


class ChannelLoRAResnet34(nn.Module):
    """ResNet34 with feature-map LoRA on every Conv2d."""

    def __init__(self, num_classes=101, r=4, alpha=8, freeze_backbone=True):
        super().__init__()
        net = torchvision.models.resnet34(
            weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        )

        if freeze_backbone:
            net.requires_grad_(False)

        self._swap_convs(net, r, alpha)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
        self.net = net

    def _swap_convs(self, module, r, alpha):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                setattr(module, name, ChannelLoRAConv2d(child, r, alpha))
            else:
                self._swap_convs(child, r, alpha)

    def forward(self, x):
        return self.net(x)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
