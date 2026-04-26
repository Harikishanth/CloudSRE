import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
mean_df = pd.read_csv(r'd:\Meta\rewardglobal_mean.csv')
best_df = pd.read_csv(r'd:\Meta\rewardglobal_best.csv')

# Extract columns
step = mean_df['Step']
mean_reward = mean_df.iloc[:, 1]
best_reward = best_df.iloc[:, 1]

# Set a beautiful, professional style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

# Create figure with high DPI for crispness
fig, ax = plt.subplots(figsize=(14, 7), dpi=300)

# Plot the raw data
ax.plot(step, mean_reward, label='Global Mean Reward (4-run group)', color='#1f77b4', linewidth=2, alpha=0.7)
ax.plot(step, best_reward, label='Global Best Reward', color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)

# Plot a smoothed trendline for the mean to show the learning curve clearly
smoothed_mean = mean_reward.rolling(window=5, min_periods=1).mean()
ax.plot(step, smoothed_mean, label='Smoothed Mean Trend (window=5)', color='#ff7f0e', linewidth=4)

# Get dynamic Y limits to place text correctly
y_min, y_max = ax.get_ylim()
y_text_pos = y_max - (y_max - y_min) * 0.05

# Background shading for curriculum tiers
ax.axvspan(0, 25.5, color='#e6f2ff', alpha=0.5, zorder=0)
ax.text(12.5, y_text_pos, 'Phase 1: Warmup Tier\n(Single isolated faults)', 
        ha='center', va='top', fontsize=13, fontweight='bold', color='#1f77b4')

ax.axvspan(25.5, 50, color='#f2e6ff', alpha=0.5, zorder=0)
ax.text(37.5, y_text_pos, 'Phase 2: Single Fault Tier\n(Faults + Misleading Red Herrings)', 
        ha='center', va='top', fontsize=13, fontweight='bold', color='#9467bd')

# Annotation for the breakthrough
# The biggest jump in mean reward happens at step 26 (-0.87 to +0.59)
ax.annotate('Policy Update Breakthrough:\nAgent learns to ignore red herrings', 
            xy=(26, 0.5975), xytext=(18, -0.2),
            arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2, headwidth=10),
            fontsize=12, ha='right', va='center',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9))

# Labels, Title, and Formatting
ax.set_title('GRPO Reinforcement Learning Reward Curve (Qwen2.5-1.5B)', 
             fontsize=18, fontweight='bold', pad=20)
ax.set_xlabel('Training Step (Episode)', fontsize=15, fontweight='bold', labelpad=15)
ax.set_ylabel('Policy Reward', fontsize=15, fontweight='bold', labelpad=15)

# Beautiful legend
ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True, borderpad=1)

# Subtle grid
ax.grid(True, linestyle=':', alpha=0.6, color='gray')
sns.despine()

# Save the plot directly over the old one
plt.tight_layout()
save_path = r'd:\Meta\cloud_sre_v2\reward_curve.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"Successfully generated breathtaking plot at: {save_path}")
