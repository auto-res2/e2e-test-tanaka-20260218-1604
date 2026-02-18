"""
Evaluation script for RC-CoT experiments.
Fetches results from WandB, exports metrics, and generates comparison plots.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb


def fetch_run_data(entity: str, project: str, run_id: str) -> Dict:
    """
    Fetch run data from WandB API.
    
    Returns:
        Dictionary with 'config', 'summary', 'history'
    """
    api = wandb.Api()
    run_path = f"{entity}/{project}/{run_id}"
    
    try:
        run = api.run(run_path)
        
        # Get config
        config = dict(run.config)
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get history (logged metrics over time)
        history = run.history()
        
        return {
            'config': config,
            'summary': summary,
            'history': history
        }
    except Exception as e:
        print(f"Warning: Could not fetch run {run_id} from WandB: {e}")
        return {
            'config': {},
            'summary': {},
            'history': pd.DataFrame()
        }


def export_per_run_metrics(results_dir: Path, run_id: str, data: Dict) -> None:
    """
    Export per-run metrics to JSON and create per-run figures.
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Export summary metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(data['summary'], f, indent=2)
    print(f"Exported metrics: {metrics_file}")
    
    # Create per-run figures
    if not data['history'].empty:
        # Accuracy over time
        if 'accuracy' in data['history'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['history']['accuracy'], linewidth=2)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Accuracy', fontsize=12)
            ax.set_title(f'Accuracy over Time - {run_id}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            output_path = run_dir / "accuracy_over_time.pdf"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Generated figure: {output_path}")
        
        # System usage over time
        if 'system1_usage' in data['history'].columns and 'system2_usage' in data['history'].columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data['history']['system1_usage'], label='System 1 Usage', linewidth=2)
            ax.plot(data['history']['system2_usage'], label='System 2 Usage', linewidth=2)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel('Usage Rate', fontsize=12)
            ax.set_title(f'System Usage over Time - {run_id}', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            output_path = run_dir / "system_usage_over_time.pdf"
            plt.savefig(output_path, bbox_inches='tight', dpi=300)
            plt.close()
            print(f"Generated figure: {output_path}")


def export_comparison_metrics(
    results_dir: Path,
    run_ids: List[str],
    all_data: Dict[str, Dict]
) -> None:
    """
    Export aggregated comparison metrics.
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics by run_id
    metrics_by_run = {}
    for run_id in run_ids:
        if run_id in all_data and all_data[run_id]['summary']:
            metrics_by_run[run_id] = all_data[run_id]['summary']
    
    # Determine primary metric
    primary_metric = "accuracy"
    
    # Compute best proposed and baseline
    proposed_runs = [rid for rid in run_ids if rid.startswith('proposed')]
    baseline_runs = [rid for rid in run_ids if rid.startswith('comparative')]
    
    best_proposed = None
    best_proposed_value = -1
    if proposed_runs:
        for rid in proposed_runs:
            if rid in metrics_by_run and primary_metric in metrics_by_run[rid]:
                value = metrics_by_run[rid][primary_metric]
                if value > best_proposed_value:
                    best_proposed_value = value
                    best_proposed = rid
    
    best_baseline = None
    best_baseline_value = -1
    if baseline_runs:
        for rid in baseline_runs:
            if rid in metrics_by_run and primary_metric in metrics_by_run[rid]:
                value = metrics_by_run[rid][primary_metric]
                if value > best_baseline_value:
                    best_baseline_value = value
                    best_baseline = rid
    
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed_value - best_baseline_value
    
    # Export aggregated metrics
    aggregated = {
        'primary_metric': primary_metric,
        'metrics_by_run': metrics_by_run,
        'best_proposed': best_proposed,
        'best_baseline': best_baseline,
        'gap': gap
    }
    
    metrics_file = comparison_dir / "aggregated_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Exported aggregated metrics: {metrics_file}")


def generate_comparison_plots(
    results_dir: Path,
    run_ids: List[str],
    all_data: Dict[str, Dict]
) -> None:
    """
    Generate comparison plots overlaying all runs.
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(run_ids))
    
    # Comparison of final accuracy
    final_accuracies = {}
    for run_id in run_ids:
        if run_id in all_data and 'accuracy' in all_data[run_id]['summary']:
            final_accuracies[run_id] = all_data[run_id]['summary']['accuracy']
    
    if final_accuracies:
        fig, ax = plt.subplots(figsize=(12, 6))
        runs = list(final_accuracies.keys())
        values = list(final_accuracies.values())
        
        bars = ax.bar(range(len(runs)), values, color=colors[:len(runs)])
        ax.set_xlabel('Run ID', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Final Accuracy Comparison', fontsize=14)
        ax.set_xticks(range(len(runs)))
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=10)
        
        output_path = comparison_dir / "comparison_accuracy.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated comparison plot: {output_path}")
    
    # Comparison of accuracy over time (line plot)
    has_history = any(
        run_id in all_data and not all_data[run_id]['history'].empty
        and 'accuracy' in all_data[run_id]['history'].columns
        for run_id in run_ids
    )
    
    if has_history:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, run_id in enumerate(run_ids):
            if run_id in all_data and not all_data[run_id]['history'].empty:
                history = all_data[run_id]['history']
                if 'accuracy' in history.columns:
                    ax.plot(history['accuracy'], label=run_id, linewidth=2, color=colors[i])
        
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Accuracy over Time - All Runs', fontsize=14)
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        output_path = comparison_dir / "comparison_accuracy_over_time.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated comparison plot: {output_path}")
    
    # System usage comparison
    system_usage_data = {}
    for run_id in run_ids:
        if run_id in all_data:
            summary = all_data[run_id]['summary']
            if 'system1_usage' in summary and 'system2_usage' in summary:
                system_usage_data[run_id] = {
                    'System 1': summary['system1_usage'],
                    'System 2': summary['system2_usage']
                }
    
    if system_usage_data:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        runs = list(system_usage_data.keys())
        system1_values = [system_usage_data[r]['System 1'] for r in runs]
        system2_values = [system_usage_data[r]['System 2'] for r in runs]
        
        x = np.arange(len(runs))
        width = 0.35
        
        ax.bar(x - width/2, system1_values, width, label='System 1', color=colors[0])
        ax.bar(x + width/2, system2_values, width, label='System 2', color=colors[1])
        
        ax.set_xlabel('Run ID', fontsize=12)
        ax.set_ylabel('Usage Rate', fontsize=12)
        ax.set_title('System Usage Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(runs, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1])
        
        output_path = comparison_dir / "comparison_system_usage.pdf"
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Generated comparison plot: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Evaluate RC-CoT experiment results")
    parser.add_argument("--results_dir", type=str, required=True, help="Results directory")
    parser.add_argument("--run_ids", type=str, required=True, help="JSON list of run IDs")
    parser.add_argument("--entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--project", type=str, default=None, help="WandB project")
    args = parser.parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    results_dir = Path(args.results_dir)
    
    # Get WandB config from environment or args
    entity = args.entity or os.environ.get("WANDB_ENTITY", "airas")
    project = args.project or os.environ.get("WANDB_PROJECT", "2026-02-18")
    
    print("="*80)
    print("RC-CoT EVALUATION")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print(f"Run IDs: {run_ids}")
    print(f"WandB: {entity}/{project}")
    
    # Fetch data for all runs
    all_data = {}
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        data = fetch_run_data(entity, project, run_id)
        all_data[run_id] = data
        
        # Export per-run metrics
        export_per_run_metrics(results_dir, run_id, data)
    
    # Export comparison metrics
    print("\nGenerating comparison metrics...")
    export_comparison_metrics(results_dir, run_ids, all_data)
    
    # Generate comparison plots
    print("\nGenerating comparison plots...")
    generate_comparison_plots(results_dir, run_ids, all_data)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"All outputs saved to: {results_dir}")


if __name__ == "__main__":
    main()
