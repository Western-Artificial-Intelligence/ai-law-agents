#!/usr/bin/env python3
"""
Generate plots for the BAILIFF paper:
1. Conviction Rates (Forest Plot)
2. Flip Rates (Waterfall Plot)
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def resolve_case_column(df: pd.DataFrame) -> str:
    """Return the case identifier column expected by plotting code."""
    for candidate in ("case_identifier", "case_id", "case"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "Input CSV must contain one of: case_identifier, case_id, case."
    )

def plot_conviction_rates(df, out_dir):
    """Plot conviction rates by case and cue."""
    case_col = resolve_case_column(df)
    plt.figure(figsize=(10, 6))
    
    # Calculate means and CIs
    summary = df.groupby([case_col, 'cue_condition'])['verdict_bin'].agg(['mean', 'count', 'std']).reset_index()
    summary['se'] = summary['std'] / (summary['count'] ** 0.5)
    summary['ci'] = 1.96 * summary['se']
    
    sns.barplot(data=df, x=case_col, y='verdict_bin', hue='cue_condition', errorbar=('ci', 95), capsize=.1)
    
    plt.title('Conviction Rates by Case and Condition')
    plt.ylabel('Conviction Probability')
    plt.xlabel('Case')
    plt.ylim(0, 1)
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.savefig(out_dir / 'conviction_rates.png', dpi=300)
    print(f"Saved {out_dir / 'conviction_rates.png'}")

def plot_flip_rates(df, out_dir):
    """Plot flip rates (consistency) by case."""
    if 'pair_id' not in df.columns:
        raise ValueError("Input CSV must contain a 'pair_id' column.")
    case_col = resolve_case_column(df)

    # Group by pair_id to find flips
    pairs = df.groupby('pair_id')
    flips = []
    
    for pid, group in pairs:
        control = group[group['cue_treatment'] == 0]['verdict_bin'] if 'cue_treatment' in group.columns else group.iloc[:1]['verdict_bin']
        treatment = group[group['cue_treatment'] == 1]['verdict_bin'] if 'cue_treatment' in group.columns else group.iloc[1:2]['verdict_bin']
        if control.empty or treatment.empty:
            continue

        flips.append({
            'case_identifier': group[case_col].iloc[0],
            'flipped': int(control.iloc[0] != treatment.iloc[0]),
        })
            
    flip_df = pd.DataFrame(flips)

    if flip_df.empty:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No complete control/treatment pairs found", ha='center', va='center')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(out_dir / 'flip_rates.png', dpi=300)
        print(f"Saved {out_dir / 'flip_rates.png'}")
        return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=flip_df, x='case_identifier', y='flipped', errorbar=('ci', 95), capsize=.1, color='salmon')
    
    plt.title('Counterfactual Flip Rates (Inconsistency)')
    plt.ylabel('Flip Rate (Lower is Better)')
    plt.xlabel('Case')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / 'flip_rates.png', dpi=300)
    print(f"Saved {out_dir / 'flip_rates.png'}")

def main():
    parser = argparse.ArgumentParser(description="Generate plots from results CSV.")
    parser.add_argument("input", type=Path, help="Path to results.csv")
    parser.add_argument("--out", type=Path, default=Path("plots"), help="Output directory for plots")
    args = parser.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    df = load_data(args.input)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    plot_conviction_rates(df, args.out)
    plot_flip_rates(df, args.out)

if __name__ == "__main__":
    main()
