# Enhanced evaluation script for cross-validation results analysis

import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
from enhanced_model import EnhancedFFGModel
from enhanced_dataset import EnhancedDementiaDataset, get_processors
import enhanced_config as config

def analyze_fold_results(results_path=None):
    """Analyze and visualize 10-fold cross-validation results"""
    results_path = Path(results_path or (Path(config.RESULTS_DIR) / "10fold_results.csv"))
    if not results_path.exists():
        print(f"Results file not found at {results_path}")
        return
    
    results_df = pd.read_csv(results_path)
    print("="*60)
    print("10-FOLD CROSS-VALIDATION ANALYSIS")
    print("="*60)
    
    metrics = ['test_accuracy', 'test_f1', 'test_precision', 'test_recall', 'test_auc', 'best_val_f1']
    print(f"{'Metric':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
    print("-"*60)
    for metric in metrics:
        if metric in results_df.columns:
            values = results_df[metric].dropna()
            if len(values) == 0:
                continue
            print(
                f"{metric:<20} {values.mean():<10.4f} {values.std():<10.4f} "
                f"{values.min():<10.4f} {values.max():<10.4f}"
            )
    
    print(f"\n{'Fold':<6} {'Test Acc':<10} {'Test F1':<10} {'Best Val F1':<12} {'AUC':<10}")
    print("-"*60)
    for _, row in results_df.iterrows():
        auc_val = row['test_auc'] if 'test_auc' in row else float('nan')
        print(f"{row['fold']:<6} {row['test_accuracy']:<10.4f} {row['test_f1']:<10.4f} {row['best_val_f1']:<12.4f} {auc_val:<10.4f}")
    
    print("\nStatistical Analysis:")
    acc_mean, acc_std = results_df['test_accuracy'].mean(), results_df['test_accuracy'].std()
    f1_mean, f1_std = results_df['test_f1'].mean(), results_df['test_f1'].std()
    print(f"Accuracy 95% CI: {acc_mean - 1.96 * acc_std:.4f} - {acc_mean + 1.96 * acc_std:.4f}")
    print(f"F1-Score 95% CI: {f1_mean - 1.96 * f1_std:.4f} - {f1_mean + 1.96 * f1_std:.4f}")
    
    if {'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp'}.issubset(results_df.columns):
        aggregate_cm = np.array([
            [results_df['cm_tn'].sum(), results_df['cm_fp'].sum()],
            [results_df['cm_fn'].sum(), results_df['cm_tp'].sum()],
        ])
        print("\nAggregate Confusion Matrix (counts):")
        print(aggregate_cm)
        plot_confusion_heatmap(aggregate_cm, save_dir=config.PLOTS_DIR)
    
    create_result_plots(results_df, save_dir=config.PLOTS_DIR)
    histories = load_training_histories()
    plot_training_curves(histories, save_dir=config.PLOTS_DIR)
    
    return results_df

def create_result_plots(results_df, save_dir=config.PLOTS_DIR):
    """Create visualization plots for cross-validation results"""
    os.makedirs(save_dir, exist_ok=True)
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('10-Fold Cross-Validation Results Analysis', fontsize=16, fontweight='bold')
    folds = results_df['fold']
    
    ax1 = axes[0, 0]
    ax1.plot(folds, results_df['test_accuracy'], 'bo-', label='Accuracy', linewidth=2, markersize=6)
    ax1.plot(folds, results_df['test_f1'], 'ro-', label='F1-Score', linewidth=2, markersize=6)
    if 'test_precision' in results_df.columns:
        ax1.plot(folds, results_df['test_precision'], 'co-', label='Precision', linewidth=2, markersize=6)
    if 'test_recall' in results_df.columns:
        ax1.plot(folds, results_df['test_recall'], 'mo-', label='Recall', linewidth=2, markersize=6)
    if 'best_val_f1' in results_df.columns:
        ax1.plot(folds, results_df['best_val_f1'], 'go-', label='Best Val F1', linewidth=2, markersize=6)
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.set_title('Per-fold Metrics')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2 = axes[0, 1]
    ax2.hist(results_df['test_accuracy'], bins=8, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(results_df['test_accuracy'].mean(), color='red', linestyle='--', label=f"Mean: {results_df['test_accuracy'].mean():.3f}")
    ax2.set_xlabel('Test Accuracy')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Accuracy Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3 = axes[1, 0]
    ax3.hist(results_df['test_f1'], bins=8, alpha=0.7, color='lightcoral', edgecolor='black')
    ax3.axvline(results_df['test_f1'].mean(), color='blue', linestyle='--', label=f"Mean: {results_df['test_f1'].mean():.3f}")
    ax3.set_xlabel('Test F1-Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('F1 Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    if 'test_auc' in results_df.columns:
        ax4.plot(folds, results_df['test_auc'], 'ko-', linewidth=2, markersize=6)
        ax4.set_ylabel('AUC')
        ax4.set_ylim(0, 1)
        ax4.set_title('ROC-AUC per Fold')
    else:
        metrics_data = [results_df['test_accuracy'], results_df['test_f1'], results_df['best_val_f1']]
        labels = ['Test Accuracy', 'Test F1', 'Best Val F1']
        box_plot = ax4.boxplot(metrics_data, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        ax4.set_ylabel('Score')
        ax4.set_title('Metric Dispersion')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / 'cross_validation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\nPlots saved to {save_dir}")


def load_training_histories(history_dir=config.HISTORY_DIR):
    """Load per-fold training histories from disk."""
    history_dir = Path(history_dir)
    if not history_dir.exists():
        return {}
    histories = {}
    for csv_path in history_dir.glob("fold_*_history.csv"):
        try:
            fold_id = int(csv_path.stem.split('_')[1])
        except (IndexError, ValueError):
            continue
        histories[fold_id] = pd.read_csv(csv_path)
    return histories


def plot_training_curves(histories, save_dir=config.PLOTS_DIR):
    """Plot aggregated training/validation curves across folds."""
    if not histories:
        print("No training history files detected; skipping training-curve plots.")
        return
    os.makedirs(save_dir, exist_ok=True)
    combined = []
    for fold, df in histories.items():
        tmp = df.copy()
        tmp['fold'] = fold
        combined.append(tmp)
    history_df = pd.concat(combined, ignore_index=True)
    summary = history_df.groupby('epoch').agg({
        'train_loss': ['mean', 'std'],
        'val_loss': ['mean', 'std'],
        'train_f1': ['mean', 'std'],
        'val_f1': ['mean', 'std'],
        'train_accuracy': ['mean', 'std'],
        'val_accuracy': ['mean', 'std'],
        'learning_rate': ['mean'],
    }).fillna(0)
    summary.columns = ['_'.join(filter(None, col)).strip('_') for col in summary.columns]
    epochs = summary.index.values
    
    def _plot_with_ci(ax, mean_key, std_key, label, color):
        mean_vals = summary.get(mean_key, pd.Series(index=summary.index, dtype=float)).to_numpy()
        std_vals = summary.get(std_key, pd.Series(index=summary.index, dtype=float)).fillna(0).to_numpy()
        ax.plot(epochs, mean_vals, color=color, label=label, linewidth=2)
        ax.fill_between(epochs, mean_vals - std_vals, mean_vals + std_vals, color=color, alpha=0.15)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    _plot_with_ci(axes[0, 0], 'train_loss_mean', 'train_loss_std', 'Train Loss', 'tab:blue')
    _plot_with_ci(axes[0, 0], 'val_loss_mean', 'val_loss_std', 'Val Loss', 'tab:orange')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    _plot_with_ci(axes[0, 1], 'train_f1_mean', 'train_f1_std', 'Train F1', 'tab:green')
    _plot_with_ci(axes[0, 1], 'val_f1_mean', 'val_f1_std', 'Val F1', 'tab:red')
    axes[0, 1].set_title('F1 Trajectories')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    _plot_with_ci(axes[1, 0], 'train_accuracy_mean', 'train_accuracy_std', 'Train Accuracy', 'tab:purple')
    _plot_with_ci(axes[1, 0], 'val_accuracy_mean', 'val_accuracy_std', 'Val Accuracy', 'tab:brown')
    axes[1, 0].set_title('Accuracy Trajectories')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    lr_mean = summary.get('learning_rate_mean')
    if lr_mean is not None:
        axes[1, 1].plot(epochs, lr_mean.values, color='tab:cyan', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    out_path = Path(save_dir) / 'training_dynamics.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training dynamics plots saved to {out_path}")


def plot_confusion_heatmap(conf_matrix, save_dir=config.PLOTS_DIR):
    """Plot normalized aggregate confusion matrix."""
    os.makedirs(save_dir, exist_ok=True)
    conf_matrix = np.array(conf_matrix, dtype=np.float32)
    row_sums = conf_matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(conf_matrix, np.maximum(row_sums, 1e-6))
    plt.figure(figsize=(5, 4))
    sns.heatmap(normalized, annot=True, fmt='.2f', cmap='Blues', cbar=False,
                xticklabels=['Control', 'Dementia'], yticklabels=['Control', 'Dementia'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Aggregate Confusion Matrix')
    out_path = Path(save_dir) / 'aggregate_confusion_matrix.png'
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.show()
    print(f"Confusion matrix heatmap saved to {out_path}")

def load_best_model(fold_num=None):
    """Load the best performing model from cross-validation"""
    
    results_path = Path(config.RESULTS_DIR) / "10fold_results.csv"
    if not results_path.exists():
        print("No results file found. Please run training first.")
        return None
    
    results_df = pd.read_csv(results_path)
    
    if fold_num is None:
        # Find the fold with best test F1 score
        best_fold = results_df.loc[results_df['test_f1'].idxmax(), 'fold']
        print(f"Loading best model from fold {best_fold}")
    else:
        best_fold = fold_num
        print(f"Loading model from fold {fold_num}")
    
    model_path = Path(config.MODELS_DIR) / f"fold_{best_fold}_best_model.pth"
    
    if model_path.exists():
        device = torch.device(config.DEVICE)
        model = EnhancedFFGModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")
        return model, best_fold
    else:
        print(f"Model file not found at {model_path}")
        return None, None

def generate_final_report():
    """Generate a comprehensive final report"""
    
    print("\\nGenerating final report...")
    
    # Analyze results
    results_df = analyze_fold_results()
    
    if results_df is not None:
        precision_line = recall_line = auc_line = ""
        if 'test_precision' in results_df.columns:
            precision_line = f"- Mean Test Precision: {results_df['test_precision'].mean():.4f} ± {results_df['test_precision'].std():.4f}\n"
        if 'test_recall' in results_df.columns:
            recall_line = f"- Mean Test Recall: {results_df['test_recall'].mean():.4f} ± {results_df['test_recall'].std():.4f}\n"
        if 'test_auc' in results_df.columns:
            auc_line = f"- Mean Test ROC-AUC: {results_df['test_auc'].mean():.4f} ± {results_df['test_auc'].std():.4f}\n"
        samples = len(EnhancedDementiaDataset(config.OUTPUT_CSV_PATH)) if os.path.exists(config.OUTPUT_CSV_PATH) else 'Unknown'
        report = f"""
ENHANCED DEMENTIA DETECTION SYSTEM - FINAL REPORT
================================================

Dataset Information:
- Total samples processed: {samples}
- Cross-validation: 10-fold (8 train : 1 validation : 1 test)

Model Architecture:
- Base models: Wav2Vec2 + BERT
- Enhanced features:
  * Prosodic (6 features): pitch, jitter, energy via librosa
  * Paralinguistic (4 features): speech rate, TTR, pauses via spaCy
- Co-attention layers: {config.NUM_ATTENTION_LAYERS}

Performance Results:
- Mean Test Accuracy: {results_df['test_accuracy'].mean():.4f} ± {results_df['test_accuracy'].std():.4f}
- Mean Test F1-Score: {results_df['test_f1'].mean():.4f} ± {results_df['test_f1'].std():.4f}
{precision_line}{recall_line}{auc_line}- Best Single Fold F1: {results_df['test_f1'].max():.4f} (Fold {results_df.loc[results_df['test_f1'].idxmax(), 'fold']})

Statistical Analysis:
- Accuracy 95% CI: [{results_df['test_accuracy'].mean() - 1.96 * results_df['test_accuracy'].std():.4f}, {results_df['test_accuracy'].mean() + 1.96 * results_df['test_accuracy'].std():.4f}]
- F1-Score 95% CI: [{results_df['test_f1'].mean() - 1.96 * results_df['test_f1'].std():.4f}, {results_df['test_f1'].mean() + 1.96 * results_df['test_f1'].std():.4f}]

Training Enhancements:
- Label smoothing={config.LABEL_SMOOTHING}, weight decay={config.WEIGHT_DECAY}
- Cosine scheduler with warmup ratio {config.SCHEDULER_WARMUP_RATIO}
- Backbone unfreezing after epoch {config.UNFREEZE_EPOCH} (last {config.UNFREEZE_ACOUSTIC_LAYERS}/{config.UNFREEZE_LINGUISTIC_LAYERS} layers)

Training Configuration:
- Batch size: {config.BATCH_SIZE}
- Learning rate: {config.LEARNING_RATE}
- Max epochs: {config.EPOCHS}
- Early stopping patience: {config.PATIENCE}
"""
        
        # Save report
        report_path = Path(config.RESULTS_DIR) / "final_report.txt"
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Final report saved to {report_path}")
        print(report)
    
    return results_df

if __name__ == "__main__":
    # Run comprehensive analysis
    generate_final_report()