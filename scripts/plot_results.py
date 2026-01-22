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

def plot_conviction_rates(df, out_dir):
    """Plot conviction rates by case and cue."""
    plt.figure(figsize=(10, 6))
    
    # Calculate means and CIs
    summary = df.groupby(['case_identifier', 'cue_condition'])['verdict_bin'].agg(['mean', 'count', 'std']).reset_index()
    summary['se'] = summary['std'] / (summary['count'] ** 0.5)
    summary['ci'] = 1.96 * summary['se']
    
    sns.barplot(data=df, x='case_identifier', y='verdict_bin', hue='cue_condition', errorbar=('ci', 95), capsize=.1)
    
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
    # Group by pair_id to find flips
    pairs = df.groupby('pair_id')
    flips = []
    
    for pid, group in pairs:
        if len(group) != 2:
            continue
        
        # Check if verdicts differ
        verdicts = group['verdict_bin'].values
        if len(set(verdicts)) > 1:
            flips.append({
                'case_identifier': group['case_identifier'].iloc[0],
                'flipped': 1
            })
        else:
            flips.append({
                'case_identifier': group['case_identifier'].iloc[0],
                'flipped': 0
            })
            
    flip_df = pd.DataFrame(flips)
    
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
