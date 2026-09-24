"""Training loss is Bernoulli plus Dice. The likelihood ratio uses only the Bernoulli term, after the weights are frozen."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class HybridLoss(nn.Module):
    """BCE-with-logits plus soft Dice, on observed pixels only.

    Roofs are about 15% of a tile. Dice stops the prediction from collapsing to empty.
    """

    def __init__(self, dice_weight: float = 1.0, eps: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(
        self,
        network_logits: torch.Tensor,
        targets: torch.Tensor,
        validity_map: torch.Tensor,
    ) -> torch.Tensor:
        observed = validity_map.to(dtype=network_logits.dtype)
        weight = observed.sum().clamp_min(1.0)
        bce = F.binary_cross_entropy_with_logits(network_logits, targets, reduction="none")
        bce = (bce * observed).sum() / weight
        probs = torch.sigmoid(network_logits) * observed
        soft = targets * observed
        intersection = (probs * soft).sum()
        denom = probs.sum() + soft.sum()
        dice = 1.0 - (2.0 * intersection + self.eps) / (denom + self.eps)
        return bce + self.dice_weight * dice


def iou_from_logits(
    network_logits: torch.Tensor,
    targets: torch.Tensor,
    validity_map: torch.Tensor,
    threshold: float,
) -> tuple[float, float, float]:
    """Hit, false roof, and missed roof counts. Unobserved pixels are ignored."""
    observed = validity_map.bool()
    pred = (torch.sigmoid(network_logits) >= threshold) & observed
    truth = (targets >= 0.5) & observed
    tp = float((pred & truth).sum())
    fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    return tp, fp, fn


def intersection_over_union(tp: float, fp: float, fn: float) -> float:
    """Zero when the prediction and the roof are both empty."""
    denom = tp + fp + fn
    if denom == 0.0:
        return 0.0
    return tp / denom


def summed_bernoulli_loglik(
    network_logits: torch.Tensor,
    targets: torch.Tensor,
    validity_map: torch.Tensor,
) -> torch.Tensor:
    """Sum of pixel log-probabilities on observed pixels. The ratio uses this sum, not the mean."""
    observed = validity_map.to(dtype=network_logits.dtype)
    nll = F.binary_cross_entropy_with_logits(network_logits, targets, reduction="none")
    return -(nll * observed).sum()


def null_logits(prior: float, network_logits: torch.Tensor) -> torch.Tensor:
    """One logit everywhere, from the null roof rate. Kept off 0 and 1 so the logit stays finite."""
    rate = min(max(prior, 1e-4), 1.0 - 1e-4)
    return torch.full_like(network_logits, math.log(rate / (1.0 - rate)))


@torch.no_grad()
def tile_likelihood_ratio(
    network_logits: torch.Tensor,
    targets: torch.Tensor,
    validity_map: torch.Tensor,
    prior: float,
    threshold: float,
) -> dict[str, float]:
    """Network log-likelihood against the constant roof rate, for one tile.

    TS is twice the difference. Dice is not included, and the pixels move together, so this is not a p-value.
    """
    network_likelihood = float(summed_bernoulli_loglik(network_logits, targets, validity_map))
    null_likelihood = float(summed_bernoulli_loglik(null_logits(prior, network_logits), targets, validity_map))
    delta = network_likelihood - null_likelihood
    tp, fp, fn = iou_from_logits(network_logits, targets, validity_map, threshold)
    return {
        "logL_H0": null_likelihood,
        "logL_H1": network_likelihood,
        "delta_logL": delta,
        "TS": 2.0 * delta,
        "iou": intersection_over_union(tp, fp, fn),
    }
