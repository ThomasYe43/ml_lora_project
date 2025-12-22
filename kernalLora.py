import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class KernelLoRAConv2d(nn.Module):
    """
    Minimal kernel-LoRA for Conv2d (decompose each h×w kernel into h×r and r×w).
    Only lora_A and lora_B are trainable; the base conv stays frozen.
    """

    def __init__(self, conv: nn.Conv2d, r: int = 4, alpha: int = 8):
        super().__init__()
        self.conv = conv
        for p in self.conv.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scale = alpha / float(r)

        oc, ic = conv.out_channels, conv.in_channels
        kh, kw = conv.kernel_size

        # A: (oc, ic, kh, r), B: (oc, ic, r, kw)
        self.lora_A = nn.Parameter(torch.zeros(oc, ic, kh, r))
        self.lora_B = nn.Parameter(torch.zeros(oc, ic, r, kw))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        oc, ic = self.conv.out_channels, self.conv.in_channels
        kh, kw = self.conv.kernel_size

        delta = torch.bmm(
            self.lora_A.view(oc * ic, kh, self.r),
            self.lora_B.view(oc * ic, self.r, kw),
        ).view(oc, ic, kh, kw)

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


class KernelLoRAResnet34(nn.Module):
    """ResNet34 with every Conv2d swapped for KernelLoRAConv2d."""

    def __init__(self, num_classes=101, r=4, alpha=8, freeze_backbone=True):
        super().__init__()
        net = torchvision.models.resnet34(weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1)

        if freeze_backbone:
            for p in net.parameters():
                p.requires_grad = False

        self._swap_convs(net, r, alpha)

        # Classifier
        in_f = net.fc.in_features
        net.fc = nn.Linear(in_f, num_classes)

        self.net = net

    def _swap_convs(self, module, r, alpha):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Conv2d):
                setattr(module, name, KernelLoRAConv2d(child, r=r, alpha=alpha))
            else:
                self._swap_convs(child, r, alpha)

    def forward(self, x):
        return self.net(x)

    def trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
