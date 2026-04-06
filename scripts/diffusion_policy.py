"""
Diffusion Policy: DDPM training + DDIM inference for robotic manipulation.

Architecture:
  - ResNet18 visual encoder (same conv1 adaptation as BCPolicy)
  - FiLM-conditioned MLP noise predictor with residual blocks
  - Sinusoidal timestep embedding added to visual+state conditioning
  - Cosine beta schedule (Nichol & Dhariwal 2021)
  - Epsilon (noise) prediction
  - DDIM deterministic inference
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class SinusoidalTimestepEmbedding(nn.Module):
    """Fixed sinusoidal embedding for diffusion timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        assert dim % 2 == 0, "dim must be even"
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float timesteps in [0, T)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[:, None].float() * freqs[None, :]  # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class FiLMResidualBlock(nn.Module):
    """Residual MLP block with FiLM conditioning (scale + shift)."""

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # FiLM: predict per-dim scale and shift from conditioning
        self.film = nn.Linear(cond_dim, hidden_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, hidden_dim), cond: (B, cond_dim)
        scale, shift = self.film(cond).chunk(2, dim=-1)
        return x + self.net(x) * (1.0 + scale) + shift


class MLPNoisePredictor(nn.Module):
    """MLP noise predictor for flat action spaces."""

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_proj = nn.Linear(action_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            FiLMResidualBlock(hidden_dim, cond_dim, dropout) for _ in range(n_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, action_dim)
        # Initialize output near zero — safe starting point since target is unit Gaussian noise
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, noisy_action: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # noisy_action: (B, action_dim), cond: (B, cond_dim)
        x = self.input_proj(noisy_action)
        for block in self.blocks:
            x = block(x, cond)
        return self.output_proj(x)


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        n_blocks: int = 4,
        T: int = 100,
        ddim_steps: int = 10,
        freeze_encoder: bool = False,
        in_channels: int = 3,
        state_dim: int = 0,
        dropout: float = 0.0,
        pred_horizon: int = 1,
    ):
        super().__init__()
        self.T = T
        self.single_action_dim = action_dim
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim * pred_horizon  # noise predictor operates on flattened chunk
        self.ddim_steps = ddim_steps

        # --- Visual encoder (same pattern as BCPolicy) ---
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        if in_channels != 3:
            if in_channels <= 0:
                raise ValueError(f"in_channels must be > 0, got {in_channels}")
            old_conv = backbone.conv1
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size, stride=old_conv.stride,
                padding=old_conv.padding, bias=False,
            )
            with torch.no_grad():
                repeat_factor = (in_channels + 2) // 3
                repeated = old_conv.weight.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                new_conv.weight.copy_(repeated * (3.0 / in_channels))
            backbone.conv1 = new_conv
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # (B, 512, 1, 1)
        encoder_dim = 512

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # --- Conditioning projection: image features + state → cond_dim ---
        self.obs_proj = nn.Sequential(
            nn.Linear(encoder_dim + state_dim, cond_dim),
            nn.Mish(),
        )

        # --- Timestep embedding → cond_dim ---
        time_emb_dim = cond_dim
        self.time_emb = SinusoidalTimestepEmbedding(time_emb_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_emb_dim, cond_dim),
            nn.Mish(),
            nn.Linear(cond_dim, cond_dim),
        )

        # --- Noise predictor ---
        self.noise_predictor = MLPNoisePredictor(self.action_dim, cond_dim, hidden_dim, n_blocks, dropout)

        # --- Cosine noise schedule ---
        self._build_schedule(T)

        # Pre-build DDIM timestep pairs: list of (t_curr, t_prev) from high to low
        step_ratio = max(T // ddim_steps, 1)
        ts = list(range(T - 1, -1, -step_ratio))[:ddim_steps]
        self._ddim_timesteps = ts  # list of ints, high → low

    def _build_schedule(self, T: int, s: float = 0.008) -> None:
        steps = torch.linspace(0, T, T + 1)
        f = torch.cos((steps / T + s) / (1.0 + s) * math.pi * 0.5) ** 2
        f = f / f[0]
        betas = (1.0 - f[1:] / f[:-1]).clamp(1e-4, 0.9999).float()
        alphas = 1.0 - betas
        acp = torch.cumprod(alphas, 0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", acp)
        self.register_buffer("sqrt_alphas_cumprod", acp.sqrt())
        self.register_buffer("sqrt_one_minus_alphas_cumprod", (1.0 - acp).sqrt())

    def _encode_obs(self, obs: torch.Tensor, state: torch.Tensor | None) -> torch.Tensor:
        """Encode image + optional state to conditioning vector (B, cond_dim)."""
        feat = self.encoder(obs).flatten(1)  # (B, 512)
        if state is not None:
            feat = torch.cat([feat, state], dim=1)
        return self.obs_proj(feat)  # (B, cond_dim)

    def forward(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass. Returns scalar diffusion loss."""
        B, device = actions.shape[0], actions.device

        obs_cond = self._encode_obs(obs, state)  # (B, cond_dim)

        t = torch.randint(0, self.T, (B,), device=device).long()
        noise = torch.randn_like(actions)
        x_t = (
            self.sqrt_alphas_cumprod[t, None] * actions
            + self.sqrt_one_minus_alphas_cumprod[t, None] * noise
        )

        t_emb = self.time_proj(self.time_emb(t.float()))  # (B, cond_dim)
        cond = obs_cond + t_emb

        noise_pred = self.noise_predictor(x_t, cond)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def get_action(
        self,
        obs: torch.Tensor,
        state: torch.Tensor | None,
    ) -> torch.Tensor:
        """DDIM deterministic inference. Returns (B, pred_horizon, single_action_dim) normalized action chunk."""
        B, device = obs.shape[0], obs.device

        obs_cond = self._encode_obs(obs, state)  # (B, cond_dim) — encode once

        x = torch.randn(B, self.action_dim, device=device)

        for i, t_curr in enumerate(self._ddim_timesteps):
            t_tensor = torch.full((B,), t_curr, device=device, dtype=torch.float32)
            t_emb = self.time_proj(self.time_emb(t_tensor))
            cond = obs_cond + t_emb

            eps_pred = self.noise_predictor(x, cond)

            acp_t = self.alphas_cumprod[t_curr]
            # Estimate clean action
            x0_pred = (x - (1.0 - acp_t).sqrt() * eps_pred) / acp_t.sqrt().clamp(min=1e-8)
            x0_pred = x0_pred.clamp(-3.0, 3.0)  # guard against early-step instability

            if i < len(self._ddim_timesteps) - 1:
                t_prev = self._ddim_timesteps[i + 1]
                acp_prev = self.alphas_cumprod[t_prev]
            else:
                acp_prev = torch.tensor(1.0, device=device)  # final step → clean

            x = acp_prev.sqrt() * x0_pred + (1.0 - acp_prev).sqrt() * eps_pred

        # Reshape flat chunk back to (B, pred_horizon, single_action_dim)
        return x.view(x.shape[0], self.pred_horizon, self.single_action_dim)
