import pandas as pd
import matplotlib.pyplot as plt


def plot_screw_analysis_trend(combined_df, result_path=None):
    """绘制螺丝刀状态趋势图（含更换信号）"""
    if combined_df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
        plt.axis('off')
        if result_path is not None:
            plt.savefig(result_path, bbox_inches='tight', dpi=150)
        plt.close()
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # X 轴使用时间
    times = pd.to_datetime(combined_df['DateTime'], errors='coerce')
    positions = combined_df['screw position']
    scores = combined_df['weighted_anomaly']
    signals = combined_df['replacement_signal']

    # 绘制加权异常得分
    scatter = ax1.scatter(times, scores, c=positions, cmap='viridis', s=30, alpha=0.8, label='Anomaly Score')
    ax1.set_ylabel('Weighted Anomaly Score')
    ax1.set_xlabel('Time')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # 突出显示更换信号点
    alert_mask = signals == 1
    if alert_mask.any():
        ax1.scatter(times[alert_mask], scores[alert_mask], color='red', s=60, edgecolors='black', linewidth=1.2, label='Replacement Alert')

    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 添加 colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Screw Position')

    plt.title('Screwdriver Health Monitoring Trend')
    plt.tight_layout()
    if result_path is not None:
        plt.savefig(result_path, dpi=150)
    plt.close()