"""
Behavior cloning policy: ResNet18 vision encoder + MLP action head.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BCPolicy(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int = 256, freeze_encoder: bool = False):
        super().__init__()

        # Vision encoder
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # Remove final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
        encoder_dim = 512

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (B, C, H, W) normalized image
        returns: (B, action_dim) predicted action
        """
        features = self.encoder(obs)          # (B, 512, 1, 1)
        features = features.flatten(1)        # (B, 512)
        return self.action_head(features)     # (B, action_dim)
