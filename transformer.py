import math
import torch
from torch import nn
import timm

class FcLoRALinear(nn.Module):
    """
    线性层 + LoRA（只训练 ΔW）：
    - 冻结原始全秩权重 W
    - 只训练低秩增量 ΔW = B @ A
    """
    def __init__(self, in_features, out_features, r=8, alpha=16.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # 原始全秩权重 W（fc_base），被冻结
        self.fc_base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.fc_base.parameters():
            p.requires_grad = False

        # LoRA 参数：A: r x in, B: out x r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        # 初始化：A 正态/Kaiming，B 全零（开始不影响输出）
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # base：冻结的原始线性层
        base = self.fc_base(x)  # [B, out_features]

        # LoRA：x @ A^T -> [B, r]，再 @ B^T -> [B, out_features]
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scaling * delta





class ViT_Baseline(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=101):
        super().__init__()
        # 预训练 ViT
        self.backbone = timm.create_model(model_name, pretrained=True)

        # 冻结 backbone 除 head 外的所有参数
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

        # 替换 head 为适配 num_classes 的线性层
        in_features = self.backbone.head.in_features  # 通常是 768
        self.backbone.head = nn.Linear(in_features, num_classes)

        print(f"ViT_Baseline created with frozen backbone, trainable head ({in_features} -> {num_classes})")

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ViT_FcLoRA(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=101, lora_r=8, lora_alpha=16.0):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)

        # 冻结所有参数（包括原始 head）
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 用 FcLoRALinear 替换 head（只训练 LoRA 参数）
        in_features = self.backbone.head.in_features
        self.backbone.head = FcLoRALinear(
            in_features=in_features,
            out_features=num_classes,
            r=lora_r,
            alpha=lora_alpha,
            bias=True
        )

        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ViT_FcLoRA created: r={lora_r}, alpha={lora_alpha}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
