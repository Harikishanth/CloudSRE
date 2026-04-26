import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Load Reward Data
reward_mean_df = pd.read_csv(r'd:\Meta\rewardglobal_mean_cascade+multicascade+adversarial.csv')
reward_best_df = pd.read_csv(r'd:\Meta\rewardglobal_best_cascade+multicascade+adversarial.csv')

# Load Resolution Data
res_casc = pd.read_csv(r'd:\Meta\resolutioncascaderate.csv')
res_multi = pd.read_csv(r'd:\Meta\resolutionmulti_cascaderate.csv')
res_adv = pd.read_csv(r'd:\Meta\resolutionadversarialrate.csv')

# Combine Resolution Data
res_casc.columns = ['Step', 'Rate', 'Min', 'Max']
res_multi.columns = ['Step', 'Rate', 'Min', 'Max']
res_adv.columns = ['Step', 'Rate', 'Min', 'Max']
res_combined = pd.concat([res_casc, res_multi, res_adv]).sort_values('Step')

# Set Style
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.family'] = 'sans-serif'

# Create Figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
fig.suptitle('Phase 2 Training: Escalating Death Spirals & Adversarial Attacks', fontsize=22, fontweight='bold', y=1.05)

# --- PLOT 1: REWARD CURVE ---
step_rw = reward_mean_df['Step']
mean_rw = reward_mean_df.iloc[:, 1]
best_rw = reward_best_df.iloc[:, 1]

ax1.plot(step_rw, mean_rw, label='Global Mean Reward', color='#1f77b4', linewidth=2, alpha=0.7)
ax1.plot(step_rw, best_rw, label='Global Best Reward', color='#2ca02c', linewidth=2, linestyle='--', alpha=0.8)
smoothed_mean = mean_rw.rolling(window=3, min_periods=1).mean()
ax1.plot(step_rw, smoothed_mean, label='Smoothed Mean (window=3)', color='#ff7f0e', linewidth=4)

ax1.set_title('GRPO Policy Reward over Time', fontsize=18, fontweight='bold', pad=15)
ax1.set_xlabel('Training Step (Episode)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Policy Reward', fontsize=14, fontweight='bold')
ax1.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# --- PLOT 2: RESOLUTION RATE ---
ax2.plot(res_combined['Step'], res_combined['Rate'], label='Resolution Rate (%)', color='#d62728', linewidth=3, marker='o')

ax2.set_title('Agent Success Rate (Resolving Outages)', fontsize=18, fontweight='bold', pad=15)
ax2.set_xlabel('Training Step (Episode)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
ax2.set_ylim(-5, 105)
ax2.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

# --- ADD CURRICULUM SHADING TO BOTH ---
y_pos_ax1 = ax1.get_ylim()[1] - (ax1.get_ylim()[1] - ax1.get_ylim()[0]) * 0.05
y_pos_ax2 = 95

for ax, y_pos in zip([ax1, ax2], [y_pos_ax1, y_pos_ax2]):
    # Tier 3: Cascade
    ax.axvspan(0, 12.5, color='#ffe6e6', alpha=0.5, zorder=0)
    ax.text(6.25, y_pos, 'Tier 3: Cascade\n(Dependency Failures)', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#d62728')
    
    # Tier 4: Multi-Cascade
    ax.axvspan(12.5, 24.5, color='#fff5e6', alpha=0.5, zorder=0)
    ax.text(18.5, y_pos, 'Tier 4: Multi-Cascade\n(Multiple Root Causes)', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#ff7f0e')
    
    # Tier 5: Adversarial
    ax.axvspan(24.5, 37, color='#e6ffe6', alpha=0.5, zorder=0)
    ax.text(30.5, y_pos, 'Tier 5: Adversarial\n(LLM-Driven Sabotage)', 
            ha='center', va='top', fontsize=11, fontweight='bold', color='#2ca02c')
            
    ax.grid(True, linestyle=':', alpha=0.6, color='gray')
    ax.set_xlim(1, 36)

sns.despine()

# Save
plt.tight_layout()
save_path = r'd:\Meta\cloud_sre_v2\reward_curve_leg2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"Plot saved to {save_path}")
