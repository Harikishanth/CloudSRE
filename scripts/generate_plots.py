import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse

def plot_wandb_csv(csv_path, output_name, title, y_label):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Assume CSV has columns like 'Step' and then metric columns
    step_col = [col for col in df.columns if 'Step' in col]
    if not step_col:
        print(f"Could not find a 'Step' column in {csv_path}")
        return
    step_col = step_col[0]

    # Find metric columns (ignore step, MIN, MAX columns)
    metric_cols = [col for col in df.columns if col != step_col and not col.endswith('__MIN') and not col.endswith('__MAX')]
    
    plt.figure(figsize=(10, 6))
    
    # Setup aesthetic colors
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, col in enumerate(metric_cols):
        # Drop NaNs for plotting
        clean_df = df[[step_col, col]].dropna()
        plt.plot(clean_df[step_col], clean_df[col], linewidth=2.5, label=col.split(' - ')[-1], color=colors[i % len(colors)])

    plt.title(title, fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Training Episodes', fontsize=12, fontweight='bold')
    plt.ylabel(y_label, fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=10)
    
    # Special annotation for the breakthrough if plotting cascade resolution
    if 'resolution' in csv_path.lower() and 'cascade' in title.lower():
        plt.annotate('Agent Discovers\nCascade Mitigation', 
                     xy=(14, 20), xytext=(5, 40),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                     fontsize=11, fontweight='bold', color='#e74c3c',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e74c3c", lw=2))

    plt.tight_layout()
    plt.savefig(f"{output_name}.png", dpi=300)
    print(f"Saved {output_name}.png successfully!")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot WandB CSV Exports")
    parser.add_argument("--resolution-csv", type=str, default="wandb_export_resolution.csv", help="Path to resolution CSV")
    parser.add_argument("--reward-csv", type=str, default="wandb_export_reward.csv", help="Path to reward CSV")
    args = parser.parse_args()

    print("Generating Professional Graphs for Judges...")
    plot_wandb_csv(args.resolution_csv, "cascade_resolution_rate", "Resolution Rate: Cascade Tier", "Resolution Rate (%)")
    plot_wandb_csv(args.reward_csv, "cascade_reward_curve", "Training Reward: Cascade Tier", "Average Reward")
    print("Done! You can now add these images to your README.md.")
