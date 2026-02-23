#!/usr/bin/env python3
"""
Generate the recommended single-series 3MT chart:
Verdict flip rate by model family.
"""
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "3mt_flip_rate_chart.png"

    families = ["Llama", "Phi", "Mistral", "DeepSeek", "Qwen7", "Qwen14"]
    flip_rates = [0.327, 0.341, 0.383, 0.275, 0.330, 0.149]
    pairs = [104, 176, 616, 80, 718, 777]

    x = range(len(families))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=220)

    bars = ax.bar(x, [v * 100 for v in flip_rates], color="#1f4e79", width=0.62)

    ax.set_title("Verdict Flip Rate by Model Family", fontsize=28, weight="bold", pad=18)
    ax.set_ylabel("Flip Rate (%)", fontsize=20)
    ax.set_ylim(0, 45)
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.tick_params(axis="y", labelsize=16)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(families, pairs)], fontsize=15)

    for bar, rate in zip(bars, flip_rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{rate * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=15,
            weight="bold",
            color="#0f172a",
        )

    ax.text(
        0.99,
        0.96,
        "Instability remains non-zero\nacross all families.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#e2e8f0", edgecolor="#334155"),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

