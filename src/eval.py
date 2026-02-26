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
    p = argparse.ArgumentParser(description="Evaluate a trained BloodMNIST classifier.")

    # Reproducibility
    p.add_argument("--seed", type=int)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction)

    # Paths
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--splits-path", type=Path)
    p.add_argument("--checkpoint", type=Path, required=True)

    # Inference
    p.add_argument("--image-size", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--num-workers", type=int)

    # W&B (optional)
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
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
    )

    configure_reproducibility(ReproducibilityConfig(seed=cfg.seed, deterministic=cfg.deterministic))

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import transforms

    from src.data import load_bloodmnist_test, load_splits
    from src.metrics import compute_metrics
    from src.modeling import build_model, get_device

    if not cfg.splits_path.exists():
        raise SystemExit(f"Missing {cfg.splits_path}.")

    splits = load_splits(cfg.splits_path)

    ckpt = torch.load(args.checkpoint, map_location="cpu")

    class_names = ckpt.get("class_names") or _get_class_names()
    num_classes = int(ckpt.get("num_classes") or len(class_names))

    mean = (0.485, 0.456, 0.406) if bool(ckpt.get("pretrained", True)) else (0.5, 0.5, 0.5)
    std = (0.229, 0.224, 0.225) if bool(ckpt.get("pretrained", True)) else (0.5, 0.5, 0.5)
    tfm = transforms.Compose(
        [
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_base = load_bloodmnist_test(cfg.data_dir, transform=tfm)
    test_ds = Subset(test_base, splits.test.indices)

    device = get_device()
    seed_torch_everything(cfg.seed, deterministic=cfg.deterministic)

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    worker_init = dataloader_worker_init_fn(cfg.seed)

    loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=worker_init,
        generator=generator,
        persistent_workers=(cfg.num_workers > 0),
    )

    model = build_model(str(ckpt.get("model", "resnet18")), num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(yb.numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)

    metrics = compute_metrics(y_true=y_true_np, y_pred=y_pred_np, num_classes=num_classes)

    use_wandb = cfg.wandb_mode != "disabled"
    if use_wandb:
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            config={"eval_checkpoint": str(args.checkpoint), **cfg.asdict()},
            mode=cfg.wandb_mode,
        )

        wandb.summary["test/accuracy"] = metrics.accuracy
        wandb.summary["test/weighted_f1"] = metrics.weighted_f1
        for i, f1 in enumerate(metrics.per_class_f1):
            wandb.summary[f"test/f1_class_{i}"] = f1

        try:
            wandb.log(
                {
                    "test/confusion_matrix": wandb.plot.confusion_matrix(
                        y_true=y_true_np.tolist(),
                        preds=y_pred_np.tolist(),
                        class_names=class_names,
                    )
                }
            )
        except Exception:
            pass

        run.finish()

    print(f"Test acc: {metrics.accuracy:.4f} | weighted F1: {metrics.weighted_f1:.4f}")


if __name__ == "__main__":
    main()
