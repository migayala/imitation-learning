"""
Behavior cloning policy: ResNet18 vision encoder + MLP action head.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class BCPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        freeze_encoder: bool = False,
        in_channels: int = 3,
        state_dim: int = 0,
    ):
        super().__init__()

        # Vision encoder
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            if in_channels <= 0:
                raise ValueError(f"in_channels must be > 0, got {in_channels}")
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                repeat_factor = (in_channels + 2) // 3
                repeated = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                scale = float(in_channels) / 3.0
                new_conv.weight.copy_(repeated / scale)
            backbone.conv1 = new_conv

        # Remove final classification layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # output: (B, 512, 1, 1)
        encoder_dim = 512

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Action head — input is image features + optional low-dim state
        self.action_head = nn.Sequential(
            nn.Linear(encoder_dim + state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, obs: torch.Tensor, state: torch.Tensor | None = None) -> torch.Tensor:
        """
        obs:   (B, C, H, W) normalized image
        state: (B, state_dim) normalized low-dim state, or None
        returns: (B, action_dim) predicted action
        """
        features = self.encoder(obs).flatten(1)   # (B, 512)
        if state is not None:
            features = torch.cat([features, state], dim=1)  # (B, 512 + state_dim)
        return self.action_head(features)          # (B, action_dim)
