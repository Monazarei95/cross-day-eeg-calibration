"""Paired conditional-negative and uniform-negative EEG pretraining."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = nn.Conv1d(channels, 32, 1, bias=False)
        self.temporal = nn.Conv1d(32, 32, 9, padding=4, groups=32, bias=False)
        self.head = nn.Sequential(nn.Linear(32, 32), nn.ELU(), nn.Linear(32, 16))

    def forward(self, x):
        filtered = self.temporal(self.spatial(x))
        return torch.log(filtered.square().mean(dim=-1) + 1e-5)


def negative_indices(desc, mode, count, uniform_fraction, rng):
    """Sample distinct negatives inside one session batch. Exclude each anchor."""
    n = len(desc)
    if not 0 < count < n:
        raise ValueError("Negative count must be smaller than batch size")
    d2 = ((desc[:, None] - desc[None, :]) ** 2).mean(axis=-1)
    off = ~np.eye(n, dtype=bool)
    uniform = off / (n - 1)
    # A narrower kernel emphasizes nearby EOG descriptors; set from the batch alone.
    bandwidth_squared = max(float(np.median(d2[off])) / 4, 1e-8)
    weights = np.exp(-d2 / (2 * bandwidth_squared)) * off
    totals = weights.sum(axis=1, keepdims=True)
    weights = np.divide(weights, totals, out=uniform.copy(), where=totals > 0)
    if mode == "uniform":
        weights = uniform
    else:
        weights = (1 - uniform_fraction) * weights + uniform_fraction * uniform
    selected = np.stack([rng.choice(n, count, replace=False, p=w) for w in weights])
    return selected


def make_view(x, config, rng):
    length = round(config["crop_fraction"] * x.shape[-1])
    width = round(config["mask_fraction"] * length)
    starts = rng.integers(0, x.shape[-1] - length + 1, size=len(x))
    views = np.stack([trial[:, start : start + length] for trial, start in zip(x, starts)])
    for trial, start in zip(views, rng.integers(0, length - width + 1, size=len(x))):
        trial[:, start : start + width] = 0
    return torch.from_numpy(views)


def train(sources, config, mode, emit):
    seed = config["seed"]
    torch.manual_seed(seed)
    order, views, negatives, shuffle = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(4)
    ]
    descriptors = {}
    for participant, data in sources.items():
        d = data["descriptor"]
        descriptors[participant] = d[shuffle.permutation(len(d))] if mode == "shuffled" else d
    net = Encoder(next(iter(sources.values()))["x"].shape[1])
    optimizer = torch.optim.Adam(net.parameters(), lr=config["learning_rate"])
    for epoch in range(config["epochs"]):
        losses, distances = [], []
        for participant in order.permutation(list(sources)):
            x = sources[participant]["x"]
            for ids in np.array_split(order.permutation(len(x)), len(x) // config["batch_size"]):
                first, second = [make_view(x[ids], config, views) for _ in range(2)]
                desc = descriptors[participant][ids]
                selected = negative_indices(
                    desc, mode, config["negatives"], config["uniform_fraction"], negatives
                )
                native = sources[participant]["descriptor"][ids]
                distances.append(float(((native[:, None] - native[selected]) ** 2).mean()))
                z, h = [F.normalize(net.head(net(view)), dim=1) for view in (first, second)]
                candidates = torch.from_numpy(np.column_stack((np.arange(len(ids)), selected)))
                similarities = z @ h.T / config["temperature"]
                positive = torch.zeros(len(ids), dtype=torch.long)
                loss = (
                    F.cross_entropy(similarities.gather(1, candidates), positive)
                    + F.cross_entropy(similarities.T.gather(1, candidates), positive)
                ) / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        emit(
            "epoch",
            arm=mode,
            epoch=epoch + 1,
            loss=np.mean(losses),
            native_negative_distance=np.mean(distances),
        )
    return net.eval()
