"""
Visualize experimental results
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """
    Create comparison plots for Baseline vs Adaptive vs Fixed-10.
    
    Args:
        df: Results dataframe
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    # Figure 1: Token usage comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar plot of average tokens
    methods = ['Baseline', 'Adaptive', 'Fixed-10']
    avg_tokens = [
        df['baseline_tokens'].mean(),
        df['adaptive_tokens'].mean(),
        df['fixed10_tokens'].mean()
    ]
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    ax1.bar(methods, avg_tokens, color=colors, alpha=0.7)
    ax1.set_ylabel('Average Tokens', fontsize=12)
    ax1.set_title('Token Usage Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(avg_tokens):
        ax1.text(i, v + 500, f'{v:.0f}', ha='center', fontweight='bold')
    
    # Success rate comparison
    success_rates = [
        df['baseline_patch_applied'].mean() * 100,
        df['adaptive_patch_applied'].mean() * 100,
        df['fixed10_patch_applied'].mean() * 100
    ]
    
    ax2.bar(methods, success_rates, color=colors, alpha=0.7)
    ax2.set_ylabel('Success Rate (%)', fontsize=12)
    ax2.set_title('Patch Application Success', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 110)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(success_rates):
        ax2.text(i, v + 2, f'{v:.0f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_overview.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/comparison_overview.png")
    plt.close()
    
    # Figure 2: Efficiency analysis (tokens per success)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    tokens_per_success = []
    for method, token_col, applied_col in [
        ('Baseline', 'baseline_tokens', 'baseline_patch_applied'),
        ('Adaptive', 'adaptive_tokens', 'adaptive_patch_applied'),
        ('Fixed-10', 'fixed10_tokens', 'fixed10_patch_applied')
    ]:
        successful = df[df[applied_col] == True]
        if len(successful) > 0:
            tokens_per_success.append({
                'Method': method,
                'Tokens/Success': successful[token_col].mean()
            })
        else:
            tokens_per_success.append({
                'Method': method,
                'Tokens/Success': df[token_col].mean()  # Use all if none successful
            })
    
    tps_df = pd.DataFrame(tokens_per_success)
    
    ax.bar(tps_df['Method'], tps_df['Tokens/Success'], color=colors, alpha=0.7)
    ax.set_ylabel('Tokens per Successful Patch', fontsize=12)
    ax.set_title('Efficiency: Tokens per Success', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, row in tps_df.iterrows():
        ax.text(i, row['Tokens/Success'] + 500, f'{row["Tokens/Success"]:.0f}', 
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/efficiency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/efficiency_comparison.png")
    plt.close()
    
    # Figure 3: Per-task breakdown
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(df))
    width = 0.25
    
    ax.bar([i - width for i in x], df['baseline_tokens'], width, 
           label='Baseline', color=colors[0], alpha=0.7)
    ax.bar([i for i in x], df['adaptive_tokens'], width,
           label='Adaptive', color=colors[1], alpha=0.7)
    ax.bar([i + width for i in x], df['fixed10_tokens'], width,
           label='Fixed-10', color=colors[2], alpha=0.7)
    
    ax.set_xlabel('Task', fontsize=12)
    ax.set_ylabel('Tokens', fontsize=12)
    ax.set_title('Token Usage per Task', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{i+1}" for i in x], rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/per_task_tokens.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir}/per_task_tokens.png")
    plt.close()
    
    # Figure 4: N allocation distribution (for adaptive)
    if 'adaptive_n' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 6))
        
        n_counts = df['adaptive_n'].value_counts().sort_index()
        
        ax.bar(n_counts.index, n_counts.values, color='#2ecc71', alpha=0.7)
        ax.set_xlabel('N Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Adaptive N Allocation Distribution', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/n_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_dir}/n_distribution.png")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize results")
    parser.add_argument("--comparison", required=True, help="Path to comparison CSV")
    parser.add_argument("--output", default="figures", help="Output directory")
    
    args = parser.parse_args()
    
    # Load data
    df = pd.read_csv(args.comparison)
    print(f"Loaded {len(df)} results from {args.comparison}")
    
    # Generate plots
    plot_comparison(df, args.output)
    
    print(f"\n✓ All plots saved to {args.output}/")


if __name__ == "__main__":
    main()