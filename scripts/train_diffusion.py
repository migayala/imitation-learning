"""
Train Diffusion Policy on robomimic demonstrations.
"""

import argparse
import hashlib
import json
import os
import random
from datetime import datetime, timezone

import torch
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import make_dataloaders
from diffusion_policy import DiffusionPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_metadata(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    print(f"Training on: {device}")

    train_loader, val_loader, action_mean_cpu, action_std_cpu, state_mean_cpu, state_std_cpu = (
        make_dataloaders(
            hdf5_path=cfg["data"]["path"],
            camera=cfg["data"]["camera"],
            image_size=cfg["data"]["image_size"],
            batch_size=cfg["training"]["batch_size"],
            train_split=cfg["data"]["train_split"],
            schema=cfg["data"].get("schema", "robomimic_image"),
            num_workers=cfg["training"].get("num_workers", 4),
            frame_stack=cfg["data"].get("frame_stack", 1),
            state_keys=cfg["data"].get("state_keys"),
            action_horizon=cfg["model"].get("pred_horizon", 1),
        )
    )

    model = DiffusionPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"].get("hidden_dim", 256),
        cond_dim=cfg["model"].get("cond_dim", 256),
        n_blocks=cfg["model"].get("n_blocks", 4),
        T=cfg["model"].get("T", 100),
        ddim_steps=cfg["model"].get("ddim_steps", 10),
        freeze_encoder=cfg["model"].get("freeze_encoder", False),
        in_channels=3 * cfg["data"].get("frame_stack", 1),
        state_dim=cfg["model"].get("state_dim", 0),
        pred_horizon=cfg["model"].get("pred_horizon", 1),
        dropout=cfg["model"].get("dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    epochs = cfg["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["training"]["log_dir"])

    config_bytes = yaml.safe_dump(cfg, sort_keys=True).encode("utf-8")
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    with open(os.path.join(cfg["training"]["checkpoint_dir"], "train_config_snapshot.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    write_metadata(
        os.path.join(cfg["training"]["checkpoint_dir"], "action_stats.json"),
        {
            "action_mean": [float(x) for x in action_mean_cpu.tolist()],
            "action_std": [float(x) for x in action_std_cpu.tolist()],
            "source": "train_split",
        },
    )
    if state_mean_cpu is not None and state_std_cpu is not None:
        write_metadata(
            os.path.join(cfg["training"]["checkpoint_dir"], "state_stats.json"),
            {
                "state_mean": [float(x) for x in state_mean_cpu.tolist()],
                "state_std": [float(x) for x in state_std_cpu.tolist()],
                "state_keys": cfg["data"].get("state_keys", []),
                "source": "train_split",
            },
        )
    write_metadata(
        os.path.join(cfg["training"]["checkpoint_dir"], "run_metadata.json"),
        {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "device": str(device),
            "seed": seed,
            "config_hash_sha256": config_hash,
        },
    )

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for obs, state, actions in tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False):
            obs = obs.to(device)
            state = state.to(device)
            actions = actions.to(device)
            state_input = state if state.shape[1] > 0 else None
            # Flatten (B, pred_horizon, action_dim) → (B, pred_horizon * action_dim)
            actions_flat = actions.view(actions.shape[0], -1)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                loss = model(obs, state_input, actions_flat)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Skip update if gradients are non-finite (can happen early with AMP)
            grad_ok = all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
            if grad_ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
                scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs, state, actions in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                obs = obs.to(device)
                state = state.to(device)
                actions = actions.to(device)
                state_input = state if state.shape[1] > 0 else None
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    loss = model(obs, state_input, actions.view(actions.shape[0], -1))
                val_loss += loss.item()
        val_loss /= len(val_loader)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("LR/lr", scheduler.get_last_lr()[0], epoch)
        print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if epoch % cfg["training"]["save_every"] == 0:
            path = os.path.join(cfg["training"]["checkpoint_dir"], f"checkpoint_ep{epoch}.pt")
            torch.save(model.state_dict(), path)
            write_metadata(
                os.path.join(cfg["training"]["checkpoint_dir"], f"checkpoint_ep{epoch}.metadata.json"),
                {
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "best_val_loss": float(min(best_val_loss, val_loss)),
                    "checkpoint_path": path,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "config_hash_sha256": config_hash,
                },
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(cfg["training"]["checkpoint_dir"], "best.pt")
            torch.save(model.state_dict(), best_path)
            write_metadata(
                os.path.join(cfg["training"]["checkpoint_dir"], "best.metadata.json"),
                {
                    "epoch": epoch,
                    "val_loss": float(val_loss),
                    "best_val_loss": float(best_val_loss),
                    "checkpoint_path": best_path,
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "config_hash_sha256": config_hash,
                },
            )

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train_diffusion.yaml")
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
