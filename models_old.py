import torch
import torch.nn as nn
import torchvision
from peft import LoraConfig, get_peft_model


class Trained_Resnet34(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.network = torchvision.models.resnet34(weights=weights)

        # Freeze all layers except final classifier
        for name, param in self.network.named_parameters():
            if not name.startswith('fc'):
                param.requires_grad = False

        # Replace final layer for classification
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)
        print("Model created with frozen backbone")

    def forward(self, x):
        return self.network(x)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LoRAResnet34(nn.Module):
    def __init__(self, num_classes=101, lora_r=8, lora_alpha=16):
        super().__init__()
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
        self.network = torchvision.models.resnet34(weights=weights)

        # Freeze all other layers
        for param in self.network.parameters():
            param.requires_grad = False

        # Replace final layer with LoRA-enhanced linear layer
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, num_classes)

        # Apply LoRA only to the classification layer
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["fc"],
            lora_dropout=0.1,
            bias="none",
            init_lora_weights=True  # A: random, B: zero
        )

        self.network = get_peft_model(self.network, lora_config)

        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LoRA Model: {trainable_params:,} trainable params ({100*trainable_params/total_params:.1f}%)")

    def forward(self, x):
        return self.network(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# import math
#
# class LoRALinear(nn.Module):
#     """
#     线性层 + LoRA：
#     - 正常的 Linear(in_features, out_features) 仍然训练
#     - 额外加一个低秩增量 BA（rank = r）
#     """
#     def __init__(self, in_features, out_features, r=8, alpha=16.0):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.r = r
#         self.alpha = alpha
#         self.scaling = alpha / r
#
#         # 原始全秩线性层（fc 本体，仍然 requires_grad=True）
#         self.fc = nn.Linear(in_features, out_features, bias=True)
#
#         # LoRA A, B：ΔW = B @ A
#         # A: r x in_features, B: out_features x r
#         self.lora_A = nn.Parameter(torch.zeros(r, in_features))
#         self.lora_B = nn.Parameter(torch.zeros(out_features, r))
#
#         # 初始化
#         nn.init.kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
#         if self.fc.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.fc.bias, -bound, bound)
#
#         nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#         nn.init.zeros_(self.lora_B)  # 开始时 LoRA 不影响输出
#
#     def forward(self, x):
#         # base: 全秩线性层
#         base = self.fc(x)  # [B, out_features]
#
#         # LoRA: x @ A^T -> [B, r]，再 @ B^T -> [B, out_features]
#         lora = (x @ self.lora_A.t()) @ self.lora_B.t()
#         return base + self.scaling * lora
#
# class LoRAResnet34(nn.Module):
#     def __init__(self, num_classes=101, lora_r=8, lora_alpha=16.0):
#         super().__init__()
#         weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
#         self.network = torchvision.models.resnet34(weights=weights)
#
#         # 冻结 backbone，只保留 fc 相关参数可训练
#         for name, param in self.network.named_parameters():
#             if not name.startswith("fc"):
#                 param.requires_grad = False
#
#         # 替换 fc 为 LoRALinear：全秩 fc + LoRA
#         num_ftrs = self.network.fc.in_features  # 512
#         self.network.fc = LoRALinear(
#             in_features=num_ftrs,
#             out_features=num_classes,
#             r=lora_r,
#             alpha=lora_alpha
#         )
#
#         trainable_params = self.get_trainable_params()
#         total_params = sum(p.numel() for p in self.parameters())
#         print("LoRAResnet34_FCPlusLoRA created:")
#         print(f"  Trainable params: {trainable_params:,} "
#               f"({100*trainable_params/total_params:.1f}%)")
#
#     def forward(self, x):
#         return self.network(x)
#
#     def get_trainable_params(self):
#         return sum(p.numel() for p in self.parameters() if p.requires_grad)


# print(Trained_Resnet34())
# print("-------------------------------------------------------------------")
# print(LoRAResnet34())