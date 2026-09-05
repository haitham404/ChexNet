"""Training and evaluation engines."""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from . import evaluate


def epoch_train(model, loader, optimizer, loss_fn):
    """One training epoch."""
    model.train()
    total_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(model.device), labels.to(model.device)
        optimizer.zero_grad()
        loss = loss_fn(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def epoch_val(model, loader, loss_fn):
    """One validation epoch."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(model.device), labels.to(model.device)
            total_loss += loss_fn(model(imgs), labels).item()
    return total_loss / len(loader)


def get_outputs(model, loader, use_tencrop: bool = False):
    """Standard inference (no TTA) or TenCrop inference."""
    model.eval()
    gts, preds = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            if use_tencrop:
                # imgs shape: (B, 10, C, H, W)
                B, n_crops, C, H, W = imgs.size()
                imgs_flat = imgs.view(-1, C, H, W).to(model.device)
                out = model(imgs_flat).view(B, n_crops, -1).mean(dim=1)
            else:
                imgs = imgs.to(model.device)
                out = model(imgs)
            gts.append(labels)
            preds.append(out.cpu())
    return torch.cat(gts), torch.cat(preds)


def compute_pos_weight(train_csv: str, device):
    """Compute class weights for handling imbalance."""
    import pandas as pd
    import numpy as np
    df = pd.read_csv(train_csv)
    label_matrix = np.vstack(df["labels"].apply(eval).values)
    pos = label_matrix.sum(axis=0)
    neg = len(label_matrix) - pos
    pos_weight = np.where(pos > 0, neg / pos, 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32).to(device)


def train(model, path_dir_data, path_train, path_val,
          arch="DENSE-NET-121", pretrained=True,
          class_count=14, batch_size=32, max_epochs=20,
          lr=1e-3, checkpoint=None):
    """Full training pipeline."""
    from .model import build_model

    model = build_model(arch, class_count, pretrained)

    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location=model.device))
        print(f"Resumed from {checkpoint}")

    num_workers = 4 if torch.cuda.is_available() else 0

    from .dataset import ChestXrayDataset
    from .transforms import get_transform_train, get_transform_val

    ds_train = ChestXrayDataset(path_train, path_dir_data, get_transform_train())
    ds_val = ChestXrayDataset(path_val, path_dir_data, get_transform_val())

    loader_train = DataLoader(ds_train, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    pos_weight = compute_pos_weight(path_train, model.device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999),
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=1)

    PATIENCE_EARLY_STOP = 3
    epochs_no_improve = 0
    best_val_loss = float("inf")

    print(f"\nTraining {arch} for up to {max_epochs} epochs on {model.device}")
    print("=" * 60)

    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        train_loss = epoch_train(model, loader_train, optimizer, loss_fn)
        val_loss = epoch_val(model, loader_val, loss_fn)
        scheduler.step(val_loss)

        gt, pred = get_outputs(model, loader_val)
        aurocs = evaluate.Metrics.compute_auroc(gt, pred, class_count)
        mean_auc = float(np.nanmean(aurocs))
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            tag = "  ← SAVED"
        else:
            epochs_no_improve += 1
            tag = f"  (no improve {epochs_no_improve}/{PATIENCE_EARLY_STOP})"

        print(f"Epoch {epoch:02d}/{max_epochs} | "
              f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
              f"ValAUC={mean_auc:.4f} | LR={current_lr:.0e} | "
              f"{elapsed/60:.1f}min{tag}")

        if epochs_no_improve >= PATIENCE_EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}.")
            break

    print(f"\nBest val loss: {best_val_loss:.4f}  →  model saved")
    return model


def test(path_dir_data, path_test, path_model="best_model.pth",
         arch="DENSE-NET-121", class_count=14,
         pretrained=True, batch_size=32, use_tencrop=True):
    """Full evaluation on test set."""
    from .model import build_model
    from .dataset import ChestXrayDataset
    from .transforms import get_transform_test_tencrop, get_transform_val
    from .evaluate import Metrics

    model = build_model(arch, class_count, pretrained)
    model.load_state_dict(torch.load(path_model, map_location=model.device))
    model.eval()

    tf = get_transform_test_tencrop() if use_tencrop else get_transform_val()
    ds = ChestXrayDataset(path_test, path_dir_data, tf)
    num_workers = 4 if torch.cuda.is_available() else 0
    loader = DataLoader(ds, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)

    if use_tencrop:
        gt, pred = get_outputs(model, loader, use_tencrop=True)
    else:
        gt, pred = get_outputs(model, loader)

    aurocs = Metrics.compute_auroc(gt, pred, class_count)
    mean_auc = float(np.nanmean(aurocs))

    print("\n" + "=" * 60)
    print("Per-class AUROC (paper targets in parentheses):")
    paper_auroc = [0.8094, 0.9248, 0.8638, 0.7345, 0.8676,
                   0.7802, 0.7680, 0.8887, 0.7901, 0.8878,
                   0.9371, 0.8047, 0.8062, 0.9164]
    for i, (cls, auc, ref) in enumerate(zip(CLASSES, aurocs, paper_auroc)):
        flag = "✓" if (not np.isnan(auc) and auc >= ref - 0.02) else "✗"
        print(f"  {flag} {cls:<20} {auc:.4f}  (paper: {ref:.4f})")

    print(f"\nMean AUROC:  {mean_auc:.4f}   (paper: 0.8417)")

    f1, thr = Metrics.compute_pneumonia_f1(gt, pred, pneumonia_idx=6)
    print(f"Pneumonia F1: {f1:.4f} @ threshold {thr:.3f}   (paper: 0.435)")

    # Bootstrap 95% CI
    boot = Metrics.bootstrap_auroc(gt, pred, class_count, n_boot=2000)
    print("\nBootstrap 95% CI (mean AUROC):")
    mean_lower = float(np.nanmean(boot["lower"]))
    mean_upper = float(np.nanmean(boot["upper"]))
    print(f"  [{mean_lower:.4f}, {mean_upper:.4f}]")

    return aurocs, mean_auc, f1