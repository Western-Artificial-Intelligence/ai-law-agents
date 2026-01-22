#!/usr/bin/env python3
"""
Generate dummy plots for the BAILIFF paper draft.
Matches the "Reverse Bias" narrative in the LaTeX file.
"""
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path

def main():
    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)
    
    # 1. Conviction Rates (Reverse Bias)
    # Data from Table in paper
    data = [
        {"Case": "Traffic", "Condition": "Control (White)", "Conviction Rate": 0.45},
        {"Case": "Traffic", "Condition": "Treatment (Non-White)", "Conviction Rate": 0.42},
        {"Case": "Assault", "Condition": "Control (White)", "Conviction Rate": 0.62},
        {"Case": "Assault", "Condition": "Treatment (Non-White)", "Conviction Rate": 0.51},
        {"Case": "Shoplifting", "Condition": "Control (White)", "Conviction Rate": 0.55},
        {"Case": "Shoplifting", "Condition": "Treatment (Non-White)", "Conviction Rate": 0.53},
        {"Case": "DUI", "Condition": "Control (White)", "Conviction Rate": 0.70},
        {"Case": "DUI", "Condition": "Treatment (Non-White)", "Conviction Rate": 0.68},
    ]
    df = pd.DataFrame(data)
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Custom palette: Blue for Control, Orange for Treatment
    sns.barplot(data=df, x="Case", y="Conviction Rate", hue="Condition", palette="muted")
    
    plt.title('Conviction Rates by Case and Condition (N=100 Pairs)')
    plt.ylabel('Conviction Probability')
    plt.ylim(0, 1.0)
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.savefig(out_dir / "conviction_rates.png", dpi=300)
    print(f"Saved {out_dir / 'conviction_rates.png'}")

    # 2. Flip Rates (Waterfall / Bar)
    # "Traffic and Shoplifting show the highest instability"
    flip_data = [
        {"Case": "Traffic", "Flip Rate": 0.12},
        {"Case": "Assault", "Flip Rate": 0.08},
        {"Case": "Shoplifting", "Flip Rate": 0.15},
        {"Case": "DUI", "Flip Rate": 0.04},
        {"Case": "Vandalism", "Flip Rate": 0.05},
        {"Case": "Petty Theft", "Flip Rate": 0.06},
    ]
    df_flip = pd.DataFrame(flip_data)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_flip, x="Case", y="Flip Rate", color="salmon")
    
    plt.title('Counterfactual Flip Rates (Consistency Gap)')
    plt.ylabel('Flip Rate (Lower is Better)')
    plt.ylim(0, 0.20)
    plt.tight_layout()
    plt.savefig(out_dir / "flip_rates.png", dpi=300)
    print(f"Saved {out_dir / 'flip_rates.png'}")

if __name__ == "__main__":
    main()
