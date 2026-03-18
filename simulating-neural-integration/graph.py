"""Deception Steering Visualization: Bar Chart with Error Bars"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load summary data
sum_file = Path("scripts/eval/sum.json")
with open(sum_file, 'r') as f:
    summary = json.load(f)

# Load raw evaluation data to calculate standard errors
def load_eval_scores(filepath, scenario_type):
    """Load scores for specific scenario type (deceptive or honest)"""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if scenario_type == "deceptive":
        # First 10 scenarios
        scenarios = data[:10]
    else:  # honest
        # Last 10 scenarios
        scenarios = data[10:20]

    # Flatten all scores from all runs
    all_scores = []
    for item in scenarios:
        if item['scores']:
            all_scores.extend(item['scores'])

    return all_scores

# Calculate means and SEs for each condition
eval_dir = Path("scripts/eval")
conditions = ['control', 'steer', 'reverse']
condition_labels = ['Control', 'Steer\n(Same Direction)', 'Reverse\n(Inverse Direction)']

deceptive_means = []
deceptive_ses = []
honest_means = []
honest_ses = []

for condition in conditions:
    # Deceptive prompts
    dec_scores = load_eval_scores(eval_dir / f"{condition if condition == 'control' else ('1' if condition == 'steer' else '2')}.json", "deceptive")
    deceptive_means.append(np.mean(dec_scores))
    deceptive_ses.append(stats.sem(dec_scores))

    # Honest prompts
    hon_scores = load_eval_scores(eval_dir / f"{condition if condition == 'control' else ('1' if condition == 'steer' else '2')}.json", "honest")
    honest_means.append(np.mean(hon_scores))
    honest_ses.append(stats.sem(hon_scores))

# Create output directory
output_dir = Path("scripts/graphs")
output_dir.mkdir(exist_ok=True)

# Set font
plt.rcParams['font.family'] = 'Helvetica'

# Colors - purple theme
deceptive_color = '#D95F8F'  # Pink/red for deceptive
honest_color = '#6A9FD9'     # Blue for honest

# X positions
x_pos = np.arange(len(condition_labels))
bar_width = 0.6

# ===== FIRST GRAPH: Deceptive Prompts =====
fig1, ax1 = plt.subplots(figsize=(10, 6))

bars1 = ax1.bar(x_pos, deceptive_means, bar_width,
                color=deceptive_color, alpha=0.7,
                yerr=deceptive_ses,
                error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

ax1.set_xticks(x_pos)
ax1.set_xticklabels(condition_labels, fontsize=12)
ax1.set_ylabel('Deception Score (1-7)', fontsize=13, fontweight='bold')
ax1.set_title('Deceptive Prompts - Activation Steering Effects', fontsize=15, fontweight='bold', pad=20)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_linewidth(1.5)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.set_ylim(0, 7)
ax1.set_yticks(range(0, 8))
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels on bars
for i, (bar, mean) in enumerate(zip(bars1, deceptive_means)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + deceptive_ses[i] + 0.15,
             f'{mean:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'deceptive_prompts.png', dpi=300, bbox_inches='tight')
print(f"Deceptive prompts graph saved to: {output_dir / 'deceptive_prompts.png'}")
plt.close()

# ===== SECOND GRAPH: Honest Prompts =====
fig2, ax2 = plt.subplots(figsize=(10, 6))

bars2 = ax2.bar(x_pos, honest_means, bar_width,
                color=honest_color, alpha=0.7,
                yerr=honest_ses,
                error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

ax2.set_xticks(x_pos)
ax2.set_xticklabels(condition_labels, fontsize=12)
ax2.set_ylabel('Deception Score (1-7)', fontsize=13, fontweight='bold')
ax2.set_title('Honest Prompts - Activation Steering Effects', fontsize=15, fontweight='bold', pad=20)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_linewidth(1.5)
ax2.spines['bottom'].set_linewidth(1.5)
ax2.set_ylim(0, 7)
ax2.set_yticks(range(0, 8))
ax2.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add value labels on bars
for i, (bar, mean) in enumerate(zip(bars2, honest_means)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + honest_ses[i] + 0.15,
             f'{mean:.2f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'honest_prompts.png', dpi=300, bbox_inches='tight')
print(f"Honest prompts graph saved to: {output_dir / 'honest_prompts.png'}")
plt.close()

# ===== Create Combined Bar Chart =====
fig2, ax = plt.subplots(figsize=(12, 7))

# X positions for grouped bars
x_pos = np.arange(len(condition_labels))
bar_width = 0.35

# Create grouped bars
bars_dec = ax.bar(x_pos - bar_width/2, deceptive_means, bar_width,
                   label='Deceptive Prompts',
                   color=deceptive_color, alpha=0.7,
                   yerr=deceptive_ses,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

bars_hon = ax.bar(x_pos + bar_width/2, honest_means, bar_width,
                   label='Honest Prompts',
                   color=honest_color, alpha=0.7,
                   yerr=honest_ses,
                   error_kw={'linewidth': 1.5, 'ecolor': 'black', 'capsize': 4})

# Formatting
ax.set_xticks(x_pos)
ax.set_xticklabels(condition_labels, fontsize=12)
ax.set_ylabel('Deception Score (1-7)', fontsize=13, fontweight='bold')
ax.set_title('Activation Steering Effects on Deception', fontsize=16, fontweight='bold', pad=20)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.set_ylim(0, 7)
ax.set_yticks(range(0, 8))
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax.legend(loc='upper left', fontsize=11, frameon=True, shadow=True)

# Add value labels
for bars in [bars_dec, bars_hon]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'deception_steering_combined.png', dpi=300, bbox_inches='tight')
print(f"Combined visualization saved to: {output_dir / 'deception_steering_combined.png'}")

# Save summary statistics
summary_data = {
    'Condition': condition_labels,
    'Deceptive_Mean': [f'{m:.3f}' for m in deceptive_means],
    'Deceptive_SE': [f'{se:.3f}' for se in deceptive_ses],
    'Honest_Mean': [f'{m:.3f}' for m in honest_means],
    'Honest_SE': [f'{se:.3f}' for se in honest_ses]
}
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(output_dir / 'summary_statistics.csv', index=False)
print(f"Summary statistics saved to: {output_dir / 'summary_statistics.csv'}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nDeceptive Prompts:")
for i, cond in enumerate(condition_labels):
    print(f"  {cond.replace(chr(10), ' ')}: {deceptive_means[i]:.3f} ± {deceptive_ses[i]:.3f}")

print("\nHonest Prompts:")
for i, cond in enumerate(condition_labels):
    print(f"  {cond.replace(chr(10), ' ')}: {honest_means[i]:.3f} ± {honest_ses[i]:.3f}")

# Calculate effect sizes
print("\n" + "="*80)
print("EFFECT SIZES (compared to Control)")
print("="*80)

print("\nDeceptive Prompts:")
print(f"  Steer: {deceptive_means[1] - deceptive_means[0]:+.3f} ({((deceptive_means[1] - deceptive_means[0])/deceptive_means[0]*100):+.1f}%)")
print(f"  Reverse: {deceptive_means[2] - deceptive_means[0]:+.3f} ({((deceptive_means[2] - deceptive_means[0])/deceptive_means[0]*100):+.1f}%)")

print("\nHonest Prompts:")
print(f"  Steer: {honest_means[1] - honest_means[0]:+.3f} ({((honest_means[1] - honest_means[0])/honest_means[0]*100):+.1f}%)")
print(f"  Reverse: {honest_means[2] - honest_means[0]:+.3f} ({((honest_means[2] - honest_means[0])/honest_means[0]*100):+.1f}%)")
