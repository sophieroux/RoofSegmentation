"""Train, score the held-out tiles, then draw the five unlabeled roofs.

Dice is only in the training loss. The hold-out score is the Bernoulli likelihood of the network against a constant roof rate, saved in selection.pt. The masks come from a second run with new weights, final.pt. That second run is not scored.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from roofseg.config import ExperimentConfig
from roofseg.data_loader import (
    ChannelStats,
    RoofTileDataset,
    channel_stats,
    list_ids,
    load_rgb,
    occupancy,
    split_ids,
)
from roofseg.model import UNet, count_parameters
from roofseg.objectives import HybridLoss, intersection_over_union, iou_from_logits, tile_likelihood_ratio
from roofseg.symmetries import DihedralBatch


def labeled_and_open(config: ExperimentConfig) -> tuple[list[str], list[str]]:
    """Ids that have a mask, and the tiles left for prediction."""
    image_ids = list_ids(config.image_dir)
    labeled = [image_id for image_id in image_ids if (config.label_dir / f"{image_id}.png").exists()]
    unlabeled = [image_id for image_id in image_ids if image_id not in set(labeled)]
    return labeled, unlabeled


def make_loader(dataset: RoofTileDataset, config: ExperimentConfig, shuffle: bool) -> DataLoader:
    """No worker processes. Each tile draws its own flip and turn."""
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=0, generator=generator)


def run_epoch(
    model: UNet,
    loader: DataLoader,
    device: torch.device,
    objective: HybridLoss,
    optimizer: torch.optim.Optimizer | None,
    threshold: float,
) -> dict[str, float]:
    """One pass. No optimizer means evaluation, with the model in eval mode."""
    training = optimizer is not None
    model.train(training)
    loss_sum = 0.0
    batches = 0
    tp = fp = fn = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        validity_map = batch["validity_map"].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            network_logits = model(images)
            loss = objective(network_logits, masks, validity_map)
            if loss.isnan().any():
                continue
            if training:
                loss.backward()
                optimizer.step()
        loss_sum += float(loss.item())
        batches += 1
        b_tp, b_fp, b_fn = iou_from_logits(network_logits.detach(), masks, validity_map, threshold)
        tp += b_tp
        fp += b_fp
        fn += b_fn
    return {"loss": loss_sum / max(batches, 1), "iou": intersection_over_union(tp, fp, fn)}


@dataclass
class FitCheckpoint:
    """Weights, plus what you need to score or predict from them."""

    model: dict[str, torch.Tensor]
    stats: dict[str, list[float]]
    base_channels: int
    best_epoch: int
    best_val_loss: float | None
    train_ids: list[str]
    val_ids: list[str]
    mask_threshold: int
    decision_threshold: float
    roof_prior: float
    per_image: list[dict[str, float | str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model"] = self.model
        return payload


def score_holdout(
    model: UNet,
    dataset: RoofTileDataset,
    device: torch.device,
    prior: float,
    threshold: float,
) -> list[dict[str, float | str]]:
    """Likelihood ratio on the restored weights.

    Batch-norm uses the saved running stats, not the stats of the single tile being scored.
    """
    model.eval()
    rows: list[dict[str, float | str]] = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            network_logits = model(batch["image"].to(device))
            stats = tile_likelihood_ratio(
                network_logits,
                batch["mask"].to(device),
                batch["validity_map"].to(device),
                prior,
                threshold,
            )
            stats["id"] = batch["id"][0]
            rows.append(stats)
    return rows


def fit_split(
    config: ExperimentConfig,
    train_ids: list[str],
    val_ids: list[str] | None,
    epochs: int,
    device: torch.device,
    prior: float,
) -> tuple[FitCheckpoint, list[dict[str, float]]]:
    """Stop when validation loss stalls. With no hold-out, train for the given number of epochs.

    On the refit the cosine clock still spans the full budget, so the learning rate matches the chosen epoch. Weight decay is reset each epoch so learning rate times weight decay stays constant. Patience starts after the warmup. A refit starts from new weights and keeps the last epoch.
    """
    stats = channel_stats(config.image_dir, train_ids, config.valid_alpha)
    train_ds = RoofTileDataset(
        config.image_dir,
        train_ids,
        stats,
        config,
        label_dir=config.label_dir,
        symmetry=DihedralBatch(config.seed).apply,
    )
    train_loader = make_loader(train_ds, config, shuffle=True)
    val_loader = None
    if val_ids:
        val_ds = RoofTileDataset(
            config.image_dir, val_ids, stats, config, label_dir=config.label_dir
        )
        val_loader = make_loader(val_ds, config, shuffle=False)

    model = UNet(in_channels=config.in_channels, base_channels=config.base_channels).to(device)
    objective = HybridLoss(dice_weight=config.dice_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    wd_product = config.learning_rate * config.weight_decay

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale = 0
    stopping_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, epochs + 1):
        train_ds.epoch = epoch
        for group in optimizer.param_groups:
            group["weight_decay"] = wd_product / group["lr"]  # VAE sync: lr * wd stays lr0 * wd0
        train_metrics = run_epoch(model, train_loader, device, objective, optimizer, config.decision_threshold)
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_iou": train_metrics["iou"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        message = (
            f"epoch {epoch}/{epochs}  train {train_metrics['loss']:.4f}  iou {train_metrics['iou']:.3f}"
        )
        if val_loader is not None:
            val_metrics = run_epoch(model, val_loader, device, objective, None, config.decision_threshold)
            row["val_loss"] = val_metrics["loss"]
            row["val_iou"] = val_metrics["iou"]
            message += f"  val {val_metrics['loss']:.4f}  iou {val_metrics['iou']:.3f}"
            if val_metrics["loss"] < best_loss - config.min_delta:
                best_loss = val_metrics["loss"]
                stale = 0
                stopping_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            elif epoch >= config.warmup_epochs:
                stale += 1
        print(
            message
            + f"  lr {optimizer.param_groups[0]['lr']:.2e}  wd {optimizer.param_groups[0]['weight_decay']:.2e}",
            flush=True,
        )
        history.append(row)
        scheduler.step()
        if val_loader is not None and epoch >= config.warmup_epochs and stale >= config.patience:
            print(f"early stop at epoch {epoch}  patience {config.patience}")
            break

    if best_state is None:
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        stopping_epoch = len(history)
    model.load_state_dict(best_state)
    if val_ids:
        print(f"restored epoch {stopping_epoch}  val loss {best_loss:.4f}")
    per_image: list[dict[str, float | str]] = []
    if val_ids:
        per_image = score_holdout(
            model,
            RoofTileDataset(config.image_dir, val_ids, stats, config, label_dir=config.label_dir),
            device,
            prior,
            config.decision_threshold,
        )
        print(f"{'id':>6} {'logL_H0':>12} {'logL_H1':>12} {'delta':>10} {'TS':>10} {'iou':>8}")
        for row_stats in per_image:
            print(
                f"{row_stats['id']:>6} {row_stats['logL_H0']:12.1f} {row_stats['logL_H1']:12.1f} "
                f"{row_stats['delta_logL']:10.1f} {row_stats['TS']:10.1f} {row_stats['iou']:8.3f}"
            )
    checkpoint = FitCheckpoint(
        model=best_state,
        stats=stats.as_dict(),
        base_channels=config.base_channels,
        best_epoch=stopping_epoch,
        best_val_loss=None if best_loss == float("inf") else best_loss,
        train_ids=train_ids,
        val_ids=list(val_ids or []),
        mask_threshold=config.mask_threshold,
        decision_threshold=config.decision_threshold,
        roof_prior=prior,
        per_image=per_image,
    )
    return checkpoint, history


def _load_checkpoint(path: Path, device: torch.device) -> FitCheckpoint | dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def write_prediction_matrix(
    config: ExperimentConfig, checkpoint_path: Path, ids: list[str], device: torch.device
) -> None:
    """One hard mask per unlabeled tile. Missing pixels stay background."""
    checkpoint = _load_checkpoint(checkpoint_path, device)
    if isinstance(checkpoint, FitCheckpoint):
        base_channels = int(checkpoint.base_channels)
        state = checkpoint.model
        stats_payload = checkpoint.stats
    else:
        base_channels = int(checkpoint["base_channels"])
        state = checkpoint["model"]
        stats_payload = checkpoint["stats"]
    model = UNet(in_channels=config.in_channels, base_channels=base_channels)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    stats = ChannelStats.from_dict(stats_payload)
    dataset = RoofTileDataset(config.image_dir, ids, stats, config, label_dir=None)
    config.prediction_dir.mkdir(parents=True, exist_ok=True)
    for index, image_id in enumerate(ids):
        batch = dataset[index]
        with torch.no_grad():
            prob = torch.sigmoid(model(batch["image"].unsqueeze(0).to(device)))[0, 0].cpu().numpy()
        _, validity_map = load_rgb(config.image_dir / f"{image_id}.png", config.valid_alpha)
        state_mask = ((prob >= config.decision_threshold) & validity_map).astype(np.uint8) * 255
        Image.fromarray(state_mask, mode="L").save(config.prediction_dir / f"{image_id}.png")
        print(f"{image_id}  roof pixels {float(state_mask.mean()) / 255.0:.1%}")


def main() -> None:
    """Pick an epoch on the hold-out, train again on all labeled tiles, write the five masks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skip-refit", action="store_true")
    parser.add_argument("--predict-only", type=Path, default=None, help="Checkpoint to score. Skips the fit.")
    args = parser.parse_args()

    overrides: dict[str, Path | int] = {}
    if args.data is not None:
        overrides["data_dir"] = args.data
    if args.out is not None:
        overrides["output_dir"] = args.out
    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    config = ExperimentConfig(**overrides)
    config.pin_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"seed {config.seed}  torch {torch.__version__}  cuda {torch.cuda.is_available()}  device {device}")

    labeled, unlabeled = labeled_and_open(config)
    if args.predict_only is not None:
        write_prediction_matrix(config, args.predict_only, unlabeled, device)
        return

    train_ids, val_ids = split_ids(labeled, config.val_fraction, config.seed)
    if len(train_ids) % config.batch_size == 1 or len(labeled) % config.batch_size == 1:
        raise SystemExit(
            f"batch size {config.batch_size} leaves a final batch of 1; BatchNorm variance is zero"
        )
    prior = occupancy(config.image_dir, config.label_dir, train_ids, config)
    print(f"labeled {len(labeled)}  train {len(train_ids)}  val {val_ids}")
    print(f"unlabeled (held out of the fit): {unlabeled}")
    print(f"H0 roof occupancy: {prior:.4f}")
    print(f"parameters: {count_parameters(UNet(base_channels=config.base_channels)):,}")

    selection, history = fit_split(config, train_ids, val_ids, config.epochs, device, prior)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selection, config.checkpoint_dir / "selection.pt")
    stopping_epoch = selection.best_epoch
    print(f"stopping epoch {stopping_epoch}  (likelihood ratio is on selection.pt)")

    final_history: list[dict[str, float]] = []
    if not args.skip_refit:
        full_prior = occupancy(config.image_dir, config.label_dir, labeled, config)
        print(
            f"refitting on all {len(labeled)} labeled tiles for {stopping_epoch} epochs "
            f"(T_max={config.epochs}, new initialization, H0 rate {full_prior:.4f})"
        )
        final, final_history = fit_split(config, labeled, None, stopping_epoch, device, full_prior)
        torch.save(final, config.checkpoint_dir / "final.pt")
        write_prediction_matrix(config, config.checkpoint_dir / "final.pt", unlabeled, device)

    report = {
        "seed": config.seed,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "unlabeled_ids": unlabeled,
        "roof_occupancy": prior,
        "mask_threshold": config.mask_threshold,
        "best_epoch": stopping_epoch,
        "best_val_loss": selection.best_val_loss,
        "per_image": selection.per_image,
        "likelihood_ratio_checkpoint": "selection.pt",
        "prediction_checkpoint": "final.pt",
        "selection_history": history,
        "final_history": final_history,
        "refit": (
            "epoch index and cosine schedule copied from the hold-out; "
            "weights re-initialized; unlabeled ids excluded; "
            "hold-out tiles re-enter only in this second fit. "
            "The likelihood ratio was not recomputed on these weights."
        ),
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {config.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
