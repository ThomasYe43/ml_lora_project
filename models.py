import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import math


class LoRAConv2d(nn.Module):
    """
    LoRA wrapper for Conv2d layers
    Adds low-rank adapters (A and B matrices) to the original conv layer
    """
    def __init__(self, conv_layer, r=8, alpha=16.0):
        super().__init__()
        self.conv = conv_layer  # Original frozen conv layer
        self.r = r
        self.alpha = alpha

        # Get conv parameters
        in_channels = conv_layer.in_channels
        out_channels = conv_layer.out_channels
        kernel_size = conv_layer.kernel_size
        stride = conv_layer.stride
        padding = conv_layer.padding

        # LoRA low-rank matrices
        # A: (r, in_channels, kernel_size, kernel_size) - reduces dimension
        # B: (out_channels, r, 1, 1) - expands dimension
        self.lora_A = nn.Parameter(torch.randn(r, in_channels, *kernel_size))
        self.lora_B = nn.Parameter(torch.zeros(out_channels, r, 1, 1))

        # Scaling factor
        self.scaling = alpha / r

        # Freeze original conv
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Original frozen convolution
        result = self.conv(x)

        # LoRA path: x -> conv(lora_A) -> conv(lora_B) -> scale
        lora_out = F.conv2d(
            x,
            self.lora_A,
            stride=self.conv.stride,
            padding=self.conv.padding
        )
        lora_out = F.conv2d(lora_out, self.lora_B)

        # Add scaled LoRA to original output
        return result + lora_out * self.scaling


# class LoRAConv2d(nn.Module):
#     """
#     1x1 通道方向 LoRA：
#     - 原始 conv_layer 冻结，保持 3x3 卷积
#     - LoRA 路径用两个 1x1 Conv2d 做通道低秩变换（in_c -> r -> out_c）
#     """
#     def __init__(self, conv_layer, r=8, alpha=16.0):
#         super().__init__()
#         self.conv = conv_layer       # 原始 3x3 conv（预训练权重）
#         self.r = r
#         self.alpha = alpha
#
#         in_channels = conv_layer.in_channels
#         out_channels = conv_layer.out_channels
#
#         # 1x1 LoRA：A: in_c -> r, B: r -> out_c
#         # A 不改变空间尺寸；B 用原 conv 的 stride/padding 来匹配输出尺寸
#         self.lora_A = nn.Conv2d(
#             in_channels,
#             r,
#             kernel_size=1,
#             stride=1,
#             padding=0,
#             bias=False
#         )
#         self.lora_B = nn.Conv2d(
#             r,
#             out_channels,
#             kernel_size=1,
#             stride=self.conv.stride,
#             padding=self.conv.padding,
#             bias=False
#         )
#
#         # 缩放因子
#         self.scaling = alpha / r
#
#         # 初始化：A 正态/Kaiming，B 全零（刚开始不影响原输出）
#         nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-3)
#         nn.init.zeros_(self.lora_B.weight)
#
#         # 冻结原始 conv
#         for param in self.conv.parameters():
#             param.requires_grad = False
#
#     def forward(self, x):
#         # 原始冻结卷积
#         base_out = self.conv(x)
#
#         # LoRA 路径：x -> A(1x1) -> B(1x1, stride/padding 跟原 conv 对齐) -> scaling
#         lora_out = self.lora_B(self.lora_A(x)) * self.scaling
#
#         # 相加得到最终输出
#         return base_out + lora_out



class Trained_Resnet34(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.network = torchvision.models.resnet34(weights=weights)

        #Freeze all layers except final classifier
        for name, param in self.network.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

        #Replace final layer for classification
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
        print("Model created with frozen backbone")

    def forward(self, x):
        return self.network(x)
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAResnet34(nn.Module):
    def __init__(self, num_classes=101, lora_r=8, lora_alpha=16.0):
        """
        Custom LoRA implementation for ResNet34
        Applies LoRA adapters to layer4's conv2 layers
        """
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.network = torchvision.models.resnet34(weights=weights)
        
        # Freeze all layers
        for param in self.network.parameters():
            param.requires_grad = False
        
        # Replace final layer for num_classes
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
        
        # Apply LoRA to layer4's conv2 in each BasicBlock
        for block in self.network.layer4:
            # Wrap conv2 with LoRA
            block.conv2 = LoRAConv2d(block.conv2, r=lora_r, alpha=lora_alpha)
        
        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LoRA Model: layer4.conv2 + FC with LoRA adapters")
        print(f"Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")

    def forward(self, x):
        return self.network(x)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

