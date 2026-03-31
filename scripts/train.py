"""
Train behavior cloning policy on robomimic demonstrations.
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import yaml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import make_dataloaders
from model import BCPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    autocast_device = "cuda" if use_amp else "cpu"
    seed = cfg["training"].get("seed", 42)
    set_seed(seed)
    print(f"Training on: {device}")
    print(f"Seed: {seed}")

    train_loader, val_loader = make_dataloaders(
        hdf5_path=cfg["data"]["path"],
        camera=cfg["data"]["camera"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["training"]["batch_size"],
        train_split=cfg["data"]["train_split"],
        schema=cfg["data"].get("schema", "robomimic_image"),
        num_workers=cfg["training"].get("num_workers", 4),
    )

    model = BCPolicy(
        action_dim=cfg["model"]["action_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        freeze_encoder=cfg["model"]["freeze_encoder"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    writer = SummaryWriter(cfg["training"]["log_dir"])
    config_snapshot = os.path.join(cfg["training"]["checkpoint_dir"], "train_config_snapshot.yaml")
    with open(config_snapshot, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    best_val_loss = float("inf")
    action_dim = cfg["model"]["action_dim"]

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        train_sqerr = torch.zeros(action_dim, dtype=torch.float64)
        for obs, actions in tqdm(train_loader, desc=f"Epoch {epoch:3d} [train]", leave=False):
            obs, actions = obs.to(device), actions.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                pred = model(obs)
                loss = criterion(pred, actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            sqerr = (pred.detach() - actions).pow(2).sum(dim=0).to(dtype=torch.float64, device="cpu")
            train_sqerr += sqerr
            train_count += actions.shape[0]
        train_loss /= len(train_loader)
        train_mse_per_dim = train_sqerr / max(train_count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        val_sqerr = torch.zeros(action_dim, dtype=torch.float64)
        with torch.no_grad():
            for obs, actions in tqdm(val_loader, desc=f"Epoch {epoch:3d} [val]", leave=False):
                obs, actions = obs.to(device), actions.to(device)
                with torch.amp.autocast(device_type=autocast_device, enabled=use_amp):
                    pred = model(obs)
                    loss = criterion(pred, actions)
                val_loss += loss.item()
                sqerr = (pred - actions).pow(2).sum(dim=0).to(dtype=torch.float64, device="cpu")
                val_sqerr += sqerr
                val_count += actions.shape[0]
        val_loss /= len(val_loader)
        val_mse_per_dim = val_sqerr / max(val_count, 1)

        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        for dim in range(action_dim):
            writer.add_scalar(f"LossPerDim/train_mse_dim{dim}", float(train_mse_per_dim[dim]), epoch)
            writer.add_scalar(f"LossPerDim/val_mse_dim{dim}", float(val_mse_per_dim[dim]), epoch)
        print(f"Epoch {epoch:3d} | train: {train_loss:.4f} | val: {val_loss:.4f}")

        if epoch % cfg["training"]["save_every"] == 0:
            path = os.path.join(cfg["training"]["checkpoint_dir"], f"checkpoint_ep{epoch}.pt")
            torch.save(model.state_dict(), path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(cfg["training"]["checkpoint_dir"], "best.pt"))

    writer.close()
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
