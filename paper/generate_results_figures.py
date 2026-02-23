#!/usr/bin/env python3
"""Generate paper-ready result figures from frozen summary metrics."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


@dataclass(frozen=True)
class RunStats:
    family: str
    run: str
    pairs: int
    b_01: int  # control 0 -> treatment 1
    c_10: int  # control 1 -> treatment 0

    @property
    def delta(self) -> float:
        return (self.b_01 - self.c_10) / self.pairs

    @property
    def flip(self) -> float:
        return (self.b_01 + self.c_10) / self.pairs


def ensure_out_dir() -> Path:
    root = Path(__file__).resolve().parent
    out = root / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_pilot_conviction_rates(out_dir: Path) -> None:
    data = [
        {"Case": "Traffic", "Condition": "Control", "Conviction Rate": 0.45},
        {"Case": "Traffic", "Condition": "Treatment", "Conviction Rate": 0.42},
        {"Case": "Simple Assault", "Condition": "Control", "Conviction Rate": 0.62},
        {"Case": "Simple Assault", "Condition": "Treatment", "Conviction Rate": 0.51},
        {"Case": "Shoplifting", "Condition": "Control", "Conviction Rate": 0.55},
        {"Case": "Shoplifting", "Condition": "Treatment", "Conviction Rate": 0.53},
        {"Case": "DUI", "Condition": "Control", "Conviction Rate": 0.70},
        {"Case": "DUI", "Condition": "Treatment", "Conviction Rate": 0.68},
    ]
    df = pd.DataFrame(data)
    order = ["Traffic", "Simple Assault", "Shoplifting", "DUI"]

    wide = df.pivot(index="Case", columns="Condition", values="Conviction Rate").reindex(order)
    x = list(range(len(order)))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(
        [i - width / 2 for i in x],
        wide["Control"].tolist(),
        width=width,
        color="#3A7CA5",
        label="Control",
    )
    ax.bar(
        [i + width / 2 for i in x],
        wide["Treatment"].tolist(),
        width=width,
        color="#F28E2B",
        label="Treatment",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Conviction Probability")
    ax.set_xlabel("Case")
    ax.set_title("Pilot Conviction Rates by Case (Llama-3-8B, 100 paired trials)")
    ax.legend(title="Cue condition", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "conviction_rates.png", dpi=300)
    plt.close(fig)


def plot_pilot_flip_rates(out_dir: Path) -> None:
    data = [
        {"Case": "Traffic", "Flip Rate": 0.12},
        {"Case": "Simple Assault", "Flip Rate": 0.08},
        {"Case": "Shoplifting", "Flip Rate": 0.15},
        {"Case": "DUI", "Flip Rate": 0.04},
        {"Case": "Vandalism", "Flip Rate": 0.05},
        {"Case": "Petty Theft", "Flip Rate": 0.06},
    ]
    df = pd.DataFrame(data)
    order = ["Traffic", "Simple Assault", "Shoplifting", "DUI", "Vandalism", "Petty Theft"]

    wide = df.set_index("Case").reindex(order)
    x = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(x, wide["Flip Rate"].tolist(), color="#E15759", width=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0.0, 0.22)
    ax.set_ylabel("Flip Rate")
    ax.set_xlabel("Case")
    ax.set_title("Pilot Counterfactual Flip Rates by Case")
    fig.tight_layout()
    fig.savefig(out_dir / "flip_rates.png", dpi=300)
    plt.close(fig)


def plot_family_effect_snapshot(out_dir: Path) -> None:
    # Counts are pooled from completed runs in the full panel.
    # b_01: control 0->treatment 1; c_10: control 1->treatment 0.
    runs = [
        RunStats("Llama-3-8B", "pooled", 104, 15, 19),
        RunStats("Phi-3", "pooled", 176, 35, 25),
        RunStats("Mistral-7B", "pooled", 616, 111, 125),
        RunStats("DeepSeek-32B", "pooled", 80, 14, 8),
        RunStats("Qwen2.5-7B", "pooled", 718, 112, 125),
        RunStats("Qwen2.5-14B", "pooled", 777, 65, 51),
    ]
    df = pd.DataFrame(
        {
            "family": [r.family for r in runs],
            "pairs": [r.pairs for r in runs],
            "delta": [r.delta for r in runs],
            "flip": [r.flip for r in runs],
        }
    )
    order = ["Llama-3-8B", "Phi-3", "Mistral-7B", "DeepSeek-32B", "Qwen2.5-7B", "Qwen2.5-14B"]
    df = df.set_index("family").reindex(order).reset_index()
    x = list(range(len(order)))

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(
        x,
        df["flip"].tolist(),
        width=0.58,
        color="#E15759",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right")
    ax.set_title("Pooled Flip Rate by Model Family")
    ax.set_ylabel("Flip rate")
    ax.set_ylim(0.0, 0.42)
    for i, p in enumerate(df["pairs"].tolist()):
        ax.text(i, df["flip"].iloc[i] + 0.01, f"n={p}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Family-Level Stability Snapshot", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "family_effect_snapshot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    out_dir = ensure_out_dir()
    plot_pilot_conviction_rates(out_dir)
    plot_pilot_flip_rates(out_dir)
    plot_family_effect_snapshot(out_dir)
    print(f"Saved figures to {out_dir}")


if __name__ == "__main__":
    main()
