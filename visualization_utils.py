import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime

def create_visualizations(metrics_without_grpo, metrics_with_grpo, power_data):
    """Create comprehensive visualizations using seaborn"""
    
    # Set up the plotting style
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (20, 24)
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 3, figsize=(20, 24))
    fig.suptitle('GRPO Model Evaluation - Comprehensive Analysis', fontsize=20, fontweight='bold')
    
    # Prepare data for seaborn plots
    
    # 1. Accuracy Metrics (Top Priority)
    ax1 = axes[0, 0]
    accuracy_data = pd.DataFrame({
        'Metric': ['Exact Match\nAccuracy', 'Partial Match\nAccuracy'] * 2,
        'Model': ['Without GRPO', 'Without GRPO', 'With GRPO', 'With GRPO'],
        'Score': [
            metrics_without_grpo['exact_match_accuracy'],
            metrics_without_grpo['partial_match_accuracy'],
            metrics_with_grpo['exact_match_accuracy'],
            metrics_with_grpo['partial_match_accuracy']
        ]
    })
    sns.barplot(data=accuracy_data, x='Metric', y='Score', hue='Model', ax=ax1)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy Score')
    ax1.set_ylim(0, 1)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.4f')
    
    # 2. F1 Score Comparison (High Priority)
    ax2 = axes[0, 1]
    f1_data = pd.DataFrame({
        'Metric': ['Macro F1', 'Micro F1'] * 2,
        'Model': ['Without GRPO', 'Without GRPO', 'With GRPO', 'With GRPO'],
        'Score': [
            metrics_without_grpo['f1_macro'],
            metrics_without_grpo['f1_micro'],
            metrics_with_grpo['f1_macro'],
            metrics_with_grpo['f1_micro']
        ]
    })
    sns.barplot(data=f1_data, x='Metric', y='Score', hue='Model', ax=ax2)
    ax2.set_title('F1 Score Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('F1 Score')
    ax2.set_ylim(0, 1)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.4f')
    
    # 3. Precision Comparison (High Priority)
    ax3 = axes[0, 2]
    precision_data = pd.DataFrame({
        'Metric': ['Macro Precision', 'Micro Precision'] * 2,
        'Model': ['Without GRPO', 'Without GRPO', 'With GRPO', 'With GRPO'],
        'Score': [
            metrics_without_grpo['precision_macro'],
            metrics_without_grpo['precision_micro'],
            metrics_with_grpo['precision_macro'],
            metrics_with_grpo['precision_micro']
        ]
    })
    sns.barplot(data=precision_data, x='Metric', y='Score', hue='Model', ax=ax3)
    ax3.set_title('Precision Comparison', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Precision Score')
    ax3.set_ylim(0, 1)
    for container in ax3.containers:
        ax3.bar_label(container, fmt='%.4f')
    
    # 4. Recall Comparison (High Priority)
    ax4 = axes[1, 0]
    recall_data = pd.DataFrame({
        'Metric': ['Macro Recall', 'Micro Recall'] * 2,
        'Model': ['Without GRPO', 'Without GRPO', 'With GRPO', 'With GRPO'],
        'Score': [
            metrics_without_grpo['recall_macro'],
            metrics_without_grpo['recall_micro'],
            metrics_with_grpo['recall_macro'],
            metrics_with_grpo['recall_micro']
        ]
    })
    sns.barplot(data=recall_data, x='Metric', y='Score', hue='Model', ax=ax4)
    ax4.set_title('Recall Comparison', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Recall Score')
    ax4.set_ylim(0, 1)
    for container in ax4.containers:
        ax4.bar_label(container, fmt='%.4f')
    
    # 5. Sample Count Comparison
    ax5 = axes[1, 1]
    sample_data = pd.DataFrame({
        'Model': ['Without GRPO', 'With GRPO'],
        'Sample Count': [metrics_without_grpo['sample_count'], metrics_with_grpo['sample_count']]
    })
    sns.barplot(data=sample_data, x='Model', y='Sample Count', ax=ax5)
    ax5.set_title('Evaluation Sample Count', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Number of Samples')
    for container in ax5.containers:
        ax5.bar_label(container, fmt='%d')
    
    # 6. Micro vs Macro Metrics Comparison
    ax6 = axes[1, 2]
    micro_macro_data = pd.DataFrame({
        'Metric Type': ['Micro F1', 'Macro F1', 'Micro Precision', 'Macro Precision', 'Micro Recall', 'Macro Recall'] * 2,
        'Model': ['Without GRPO'] * 6 + ['With GRPO'] * 6,
        'Score': [
            metrics_without_grpo['f1_micro'], metrics_without_grpo['f1_macro'],
            metrics_without_grpo['precision_micro'], metrics_without_grpo['precision_macro'],
            metrics_without_grpo['recall_micro'], metrics_without_grpo['recall_macro'],
            metrics_with_grpo['f1_micro'], metrics_with_grpo['f1_macro'],
            metrics_with_grpo['precision_micro'], metrics_with_grpo['precision_macro'],
            metrics_with_grpo['recall_micro'], metrics_with_grpo['recall_macro']
        ]
    })
    sns.barplot(data=micro_macro_data, x='Metric Type', y='Score', hue='Model', ax=ax6)
    ax6.set_title('Micro vs Macro Metrics', fontsize=14, fontweight='bold')
    ax6.set_ylabel('Score')
    ax6.tick_params(axis='x', rotation=45)
    for container in ax6.containers:
        ax6.bar_label(container, fmt='%.3f')
    
    # 7. Improvement Heatmap
    ax7 = axes[2, 0]
    improvement_data = {
        'Exact Match Accuracy': metrics_with_grpo['exact_match_accuracy'] - metrics_without_grpo['exact_match_accuracy'],
        'Partial Match Accuracy': metrics_with_grpo['partial_match_accuracy'] - metrics_without_grpo['partial_match_accuracy'],
        'Macro F1': metrics_with_grpo['f1_macro'] - metrics_without_grpo['f1_macro'],
        'Macro Precision': metrics_with_grpo['precision_macro'] - metrics_without_grpo['precision_macro'],
        'Macro Recall': metrics_with_grpo['recall_macro'] - metrics_without_grpo['recall_macro']
    }
    improvement_df = pd.DataFrame([improvement_data])
    sns.heatmap(improvement_df, annot=True, fmt='.4f', cmap='RdYlGn', center=0, ax=ax7, cbar_kws={'label': 'Improvement'})
    ax7.set_title('GRPO Improvement Heatmap', fontsize=14, fontweight='bold')
    ax7.set_ylabel('')
    
    # 8. Power Consumption - CPU and GPU
    ax8 = axes[2, 1]
    if power_data['timestamps']:
        power_df = pd.DataFrame({
            'Time': power_data['timestamps'] * 3,  # Repeat timestamps for each metric
            'Usage (%)': power_data['cpu_percent'] + power_data['gpu_utilization'] + power_data['gpu_memory'],
            'Type': ['CPU'] * len(power_data['timestamps']) + ['GPU Util'] * len(power_data['timestamps']) + ['GPU Memory'] * len(power_data['timestamps'])
        })
        sns.lineplot(data=power_df, x='Time', y='Usage (%)', hue='Type', ax=ax8)
        ax8.set_title('Power Consumption During Training', fontsize=14, fontweight='bold')
        ax8.tick_params(axis='x', rotation=45)
    else:
        ax8.text(0.5, 0.5, 'No Power Data Available', ha='center', va='center', transform=ax8.transAxes, fontsize=12)
        ax8.set_title('Power Consumption', fontsize=14, fontweight='bold')
    
    # 9. Overall Performance Radar Chart (Alternative visualization)
    ax9 = axes[2, 2]
    metrics_for_radar = ['exact_match_accuracy', 'partial_match_accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
    without_values = [metrics_without_grpo[m] for m in metrics_for_radar]
    with_values = [metrics_with_grpo[m] for m in metrics_for_radar]
    
    # Create a simple line plot as radar alternative
    x_pos = range(len(metrics_for_radar))
    ax9.plot(x_pos, without_values, 'o-', label='Without GRPO', linewidth=2, markersize=8)
    ax9.plot(x_pos, with_values, 's-', label='With GRPO', linewidth=2, markersize=8)
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels([m.replace('_', '\n').title() for m in metrics_for_radar], rotation=45, ha='right')
    ax9.set_ylabel('Score')
    ax9.set_title('Overall Performance Profile', fontsize=14, fontweight='bold')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim(0, 1)
    
    # 10. Model Performance Comparison Bar Chart
    ax10 = axes[3, 0]
    performance_data = pd.DataFrame({
        'Metric': ['Exact Match', 'Partial Match', 'Macro F1', 'Macro Precision', 'Macro Recall'] * 2,
        'Model': ['Without GRPO'] * 5 + ['With GRPO'] * 5,
        'Score': [
            metrics_without_grpo['exact_match_accuracy'], metrics_without_grpo['partial_match_accuracy'],
            metrics_without_grpo['f1_macro'], metrics_without_grpo['precision_macro'], metrics_without_grpo['recall_macro'],
            metrics_with_grpo['exact_match_accuracy'], metrics_with_grpo['partial_match_accuracy'],
            metrics_with_grpo['f1_macro'], metrics_with_grpo['precision_macro'], metrics_with_grpo['recall_macro']
        ]
    })
    sns.barplot(data=performance_data, x='Metric', y='Score', hue='Model', ax=ax10)
    ax10.set_title('Comprehensive Performance Comparison', fontsize=14, fontweight='bold')
    ax10.set_ylabel('Score')
    ax10.tick_params(axis='x', rotation=45)
    
    # 11. Performance Summary Table
    ax11 = axes[3, 1]
    ax11.axis('tight')
    ax11.axis('off')
    
    summary_table_data = [
        ['Metric', 'Without GRPO', 'With GRPO', 'Improvement'],
        ['Exact Match Acc.', f"{metrics_without_grpo['exact_match_accuracy']:.4f}", 
         f"{metrics_with_grpo['exact_match_accuracy']:.4f}", 
         f"{metrics_with_grpo['exact_match_accuracy'] - metrics_without_grpo['exact_match_accuracy']:.4f}"],
        ['Partial Match Acc.', f"{metrics_without_grpo['partial_match_accuracy']:.4f}", 
         f"{metrics_with_grpo['partial_match_accuracy']:.4f}", 
         f"{metrics_with_grpo['partial_match_accuracy'] - metrics_without_grpo['partial_match_accuracy']:.4f}"],
        ['Macro F1', f"{metrics_without_grpo['f1_macro']:.4f}", 
         f"{metrics_with_grpo['f1_macro']:.4f}", 
         f"{metrics_with_grpo['f1_macro'] - metrics_without_grpo['f1_macro']:.4f}"],
        ['Macro Precision', f"{metrics_without_grpo['precision_macro']:.4f}", 
         f"{metrics_with_grpo['precision_macro']:.4f}", 
         f"{metrics_with_grpo['precision_macro'] - metrics_without_grpo['precision_macro']:.4f}"],
        ['Macro Recall', f"{metrics_without_grpo['recall_macro']:.4f}", 
         f"{metrics_with_grpo['recall_macro']:.4f}", 
         f"{metrics_with_grpo['recall_macro'] - metrics_without_grpo['recall_macro']:.4f}"]
    ]
    
    table = ax11.table(cellText=summary_table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    # Color code the improvement column
    for i in range(1, len(summary_table_data)):
        improvement_val = float(summary_table_data[i][3])
        if improvement_val > 0:
            table[(i, 3)].set_facecolor('#90EE90')  # Light green for positive
        elif improvement_val < 0:
            table[(i, 3)].set_facecolor('#FFB6C1')  # Light red for negative
    
    ax11.set_title('Performance Summary Table', fontsize=14, fontweight='bold', pad=20)
    
    # 12. Performance Improvement Bar Chart
    ax12 = axes[3, 2]
    improvement_metrics = ['Exact Match Accuracy', 'Partial Match Accuracy', 'Macro F1', 'Macro Precision', 'Macro Recall']
    improvement_values = [
        metrics_with_grpo['exact_match_accuracy'] - metrics_without_grpo['exact_match_accuracy'],
        metrics_with_grpo['partial_match_accuracy'] - metrics_without_grpo['partial_match_accuracy'],
        metrics_with_grpo['f1_macro'] - metrics_without_grpo['f1_macro'],
        metrics_with_grpo['precision_macro'] - metrics_without_grpo['precision_macro'],
        metrics_with_grpo['recall_macro'] - metrics_without_grpo['recall_macro']
    ]
    
    colors = ['green' if val > 0 else 'red' for val in improvement_values]
    bars = ax12.bar(improvement_metrics, improvement_values, color=colors, alpha=0.7)
    ax12.set_title('GRPO Performance Improvements', fontsize=14, fontweight='bold')
    ax12.set_ylabel('Improvement Score')
    ax12.tick_params(axis='x', rotation=45)
    ax12.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, improvement_values):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    
    # Save the comprehensive plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'grpo_evaluation_comprehensive_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nComprehensive visualization saved as: {filename}")
    
    plt.show()
    
    return filename