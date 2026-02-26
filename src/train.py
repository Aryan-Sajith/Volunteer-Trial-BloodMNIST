from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import Config, with_overrides
from src.reproducibility import (
    ReproducibilityConfig,
    configure_reproducibility,
    dataloader_worker_init_fn,
    seed_torch_everything,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a BloodMNIST classifier with reproducible settings and W&B logging.")

    # Reproducibility
    p.add_argument("--seed", type=int)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction)

    # Paths
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--splits-path", type=Path)
    p.add_argument("--output-dir", type=Path)

    # Training
    p.add_argument("--model", type=str)
    p.add_argument("--pretrained", action=argparse.BooleanOptionalAction)
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight-decay", type=float)

    # W&B
    p.add_argument("--wandb-project", type=str)
    p.add_argument("--wandb-entity", type=str)
    p.add_argument("--wandb-mode", type=str)

    return p.parse_args()


def _get_class_names() -> list[str]:
    from medmnist import INFO

    labels = INFO["bloodmnist"]["label"]
    if isinstance(labels, dict):
        return [labels[str(i)] for i in range(len(labels))]
    return list(labels)


def main() -> None:
    load_dotenv()

    args = _parse_args()
    cfg = with_overrides(
        Config(),
        seed=args.seed,
        deterministic=args.deterministic,
        data_dir=args.data_dir,
        splits_path=args.splits_path,
        output_dir=args.output_dir,
        model=args.model,
        pretrained=args.pretrained,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )

    # Must happen before torch is imported/used.
    configure_reproducibility(ReproducibilityConfig(seed=cfg.seed, deterministic=cfg.deterministic))

    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms
    from tqdm import tqdm

    from src.data import load_bloodmnist_test, load_bloodmnist_trainval, load_splits
    from src.metrics import compute_metrics
    from src.modeling import build_model, get_device

    if not cfg.splits_path.exists():
        raise SystemExit(
            f"Missing {cfg.splits_path}. Generate and commit it once via: "
            "python -m src.make_splits"
        )

    splits = load_splits(cfg.splits_path)

    class_names = _get_class_names()
    num_classes = len(class_names)

    mean = (0.485, 0.456, 0.406) if cfg.pretrained else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if cfg.pretrained else (0.5, 0.5, 0.5)
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    trainval_ds = load_bloodmnist_trainval(cfg.data_dir, transform=tfm)
    test_base = load_bloodmnist_test(cfg.data_dir, transform=tfm)

    train_ds = Subset(trainval_ds, splits.train.indices)
    val_ds = Subset(trainval_ds, splits.val.indices)
    test_ds = Subset(test_base, splits.test.indices)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = cfg.output_dir / "best_model.pth"

    device = get_device()

    seed_torch_everything(cfg.seed, deterministic=cfg.deterministic)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    worker_init = dataloader_worker_init_fn(cfg.seed)

    def _loader(ds, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            pin_memory=(device.type == "cuda"),
            worker_init_fn=worker_init,
            generator=generator,
            persistent_workers=(cfg.num_workers > 0),
        )

    train_loader = _loader(train_ds, shuffle=True)
    val_loader = _loader(val_ds, shuffle=False)
    test_loader = _loader(test_ds, shuffle=False)

    model = build_model(cfg.model, num_classes=num_classes, pretrained=cfg.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    use_wandb = cfg.wandb_mode != "disabled"
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config=cfg.asdict(),
            mode=cfg.wandb_mode,
        )

    def _eval(loader: DataLoader):
        model.eval()
        losses = []
        y_true: list[int] = []
        y_pred: list[int] = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                losses.append(float(loss.item()))

                preds = torch.argmax(logits, dim=1)
                y_true.extend(yb.detach().cpu().numpy().tolist())
                y_pred.extend(preds.detach().cpu().numpy().tolist())

        y_true_np = np.asarray(y_true)
        y_pred_np = np.asarray(y_pred)
        acc = float((y_true_np == y_pred_np).mean())
        return float(np.mean(losses)) if losses else 0.0, acc, y_true_np, y_pred_np

    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        train_losses = []

        for xb, yb in tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss, val_acc, _, _ = _eval(val_loader)

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                }
            )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model": cfg.model,
                    "pretrained": cfg.pretrained,
                    "num_classes": num_classes,
                    "class_names": class_names,
                    "state_dict": model.state_dict(),
                    "config": cfg.asdict(),
                },
                best_ckpt_path,
            )

    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])

    test_loss, _, y_true, y_pred = _eval(test_loader)
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred, num_classes=num_classes)

    if use_wandb:
        import wandb

        wandb.summary["test/loss"] = test_loss
        wandb.summary["test/accuracy"] = metrics.accuracy
        wandb.summary["test/weighted_f1"] = metrics.weighted_f1
        for i, f1 in enumerate(metrics.per_class_f1):
            wandb.summary[f"test/f1_class_{i}"] = f1

        try:
            wandb.log(
                {
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true.tolist(),
                        preds=y_pred.tolist(),
                        class_names=class_names,
                    )
                }
            )
        except Exception:
            pass

        artifact = wandb.Artifact(
            name="best-model",
            type="model",
            metadata={
                "model": cfg.model,
                "pretrained": cfg.pretrained,
                "seed": cfg.seed,
                "best_val_accuracy": best_val_acc,
            },
        )
        artifact.add_file(str(best_ckpt_path))
        wandb.log_artifact(artifact)

        if wandb_run is not None:
            wandb_run.finish()

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Test acc: {metrics.accuracy:.4f} | weighted F1: {metrics.weighted_f1:.4f}")
    print(f"Saved best checkpoint to: {best_ckpt_path}")


if __name__ == "__main__":
    main()
