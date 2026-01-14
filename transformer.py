import math
import torch
from torch import nn
import timm
import torch.nn.functional as F

class FcLoRALinear(nn.Module):

    def __init__(self, in_features, out_features, r=8, alpha=16.0, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # freeze W
        self.fc_base = nn.Linear(in_features, out_features, bias=bias)
        for p in self.fc_base.parameters():
            p.requires_grad = False

        # lora：A: r x in, B: out x r
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

        
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # base
        base = self.fc_base(x)  # [B, out_features]

        # LoRA：x @ A^T -> [B, r]， @ B^T -> [B, out_features]
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()
        return base + self.scaling * delta




class ViT_Baseline(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=101):
        super().__init__()
        # pretrained ViT
        self.backbone = timm.create_model(model_name, pretrained=True)

        # freeze except head
        for name, param in self.backbone.named_parameters():
            if not name.startswith("head"):
                param.requires_grad = False

        # replace head with num_classes 
        in_features = self.backbone.head.in_features  
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


        for param in self.backbone.parameters():
            param.requires_grad = False

        # replace head with FcLoRALinear 
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


class LoRALinear(nn.Module):
    """Wrap a pretrained nn.Linear with LoRA (freeze base, train only A/B)."""
    def __init__(self, base_linear: nn.Linear, r=8, alpha=16.0, dropout=0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        self.base = base_linear
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f = base_linear.in_features
        out_f = base_linear.out_features

        # A: (r, in_f), B: (out_f, r)
        self.lora_A = nn.Parameter(torch.zeros(r, in_f))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base = self.base(x)
        x = self.dropout(x)
        delta = F.linear(F.linear(x, self.lora_A), self.lora_B)
        return base + self.scaling * delta


class ViT_AttnLoRA_HeadFT(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", num_classes=101,
                 lora_r=8, lora_alpha=16.0, lora_dropout=0.0,
                 target=("qkv", "proj"), last_n_blocks=None):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)

        # freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # trainable head (full fine-tune on head)
        in_features = self.backbone.head.in_features  # vit-b/16 usually 768
        self.backbone.head = nn.Linear(in_features, num_classes)

        # choose which blocks to apply LoRA
        blocks = list(self.backbone.blocks)
        if last_n_blocks is not None:
            blocks = blocks[-last_n_blocks:]  # e.g., only last 4 blocks

        # inject LoRA into attention linear layers
        for blk in blocks:
            if "qkv" in target:
                blk.attn.qkv = LoRALinear(blk.attn.qkv, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
            if "proj" in target:
                blk.attn.proj = LoRALinear(blk.attn.proj, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)

        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ViT_AttnLoRA_HeadFT: target={target}, r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}, last_n_blocks={last_n_blocks}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ViT_AttnFT_HeadFT(nn.Module):
    """Head full fine-tuning + Attention full fine-tuning (unfreeze qkv/proj) on last N blocks."""
    def __init__(self, model_name="vit_base_patch16_224", num_classes=101,
                 target=("qkv", "proj"), last_n_blocks=4):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True)

        # freeze everything first
        for p in self.backbone.parameters():
            p.requires_grad = False

        # trainable head
        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Linear(in_features, num_classes)  # new head is trainable by default

        blocks = list(self.backbone.blocks)
        if last_n_blocks is not None:
            blocks = blocks[-last_n_blocks:]

        # unfreeze attention linears
        for blk in blocks:
            if "qkv" in target:
                for p in blk.attn.qkv.parameters():
                    p.requires_grad = True
            if "proj" in target:
                for p in blk.attn.proj.parameters():
                    p.requires_grad = True

        trainable_params = self.get_trainable_params()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"ViT_AttnFT_HeadFT: target={target}, last_n_blocks={last_n_blocks}")
        print(f"  Trainable params: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, x):
        return self.backbone(x)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
