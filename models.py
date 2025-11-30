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
        
        # Freeze all layers
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

