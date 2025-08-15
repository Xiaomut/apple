import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates


def plot_screw_analysis_trend(combined_df, result_path=None):
    """绘制螺丝刀状态趋势图（含更换信号），优化用于四种螺丝位置 [4, 1, 2, 41]"""
    if combined_df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
        plt.axis('off')
        if result_path is not None:
            plt.savefig(result_path, bbox_inches='tight', dpi=150)
        plt.close()
        return

    # 定义螺丝位置及其对应的颜色和标签
    screw_positions = {
        1: {'color': 'blue', 'label': 'Screw Position 1'},
        2: {'color': 'green', 'label': 'Screw Position 2'},
        4: {'color': 'orange', 'label': 'Screw Position 4'},
        41: {'color': 'red', 'label': 'Screw Position 41'}
    }

    # 确保 'DateTime' 列是 datetime 类型
    combined_df['DateTime'] = pd.to_datetime(combined_df['DateTime'], errors='coerce')
    combined_df = combined_df.dropna(subset=['DateTime'])  # 删除无效的时间数据

    if combined_df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, 'No valid datetime data to display', ha='center', va='center', fontsize=12)
        plt.axis('off')
        if result_path is not None:
            plt.savefig(result_path, bbox_inches='tight', dpi=150)
        plt.close()
        return 

    times = combined_df['DateTime']
    positions = combined_df['screw position']
    scores = combined_df['weighted_anomaly']
    signals = combined_df['replacement_signal']

    # 创建图表
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 绘制每个螺丝位置的数据点
    for pos, details in screw_positions.items():
        mask = positions == pos
        if not mask.any():
            continue  # 如果该位置没有数据，跳过

        pos_times = times[mask]
        pos_scores = scores[mask]
        pos_signals = signals[mask]

        # 绘制正常数据点
        normal_mask = pos_signals == 0
        ax1.scatter(
            pos_times[normal_mask],
            pos_scores[normal_mask],
            c=[details['color']],
            s=30,
            alpha=0.6,
            label=f'{details["label"]} (Normal)',
            edgecolors='w',
            linewidth=0.5
        )

        # 绘制更换信号点
        alert_mask = pos_signals == 1
        if alert_mask.any():
            ax1.scatter(
                pos_times[alert_mask],
                pos_scores[alert_mask],
                c=[details['color']],
                s=60,
                alpha=0.9,
                label=f'{details["label"]} (Alert)',
                edgecolors='black',
                linewidth=1.2,
                marker='X'  # 使用不同的标记样式
            )

    # 设置轴标签和标题
    ax1.set_ylabel('Weighted Anomaly Score', fontsize=12)
    ax1.set_xlabel('Time', fontsize=12)
    plt.title('Screwdriver Health Monitoring Trend (Positions: 1, 2, 4, 41)', fontsize=14)

    # 格式化时间轴
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # 添加网格
    ax1.grid(True, alpha=0.3)

    # 构建图例
    # 手动创建图例项以避免重复
    legend_elements = []
    for pos, details in screw_positions.items():
        # 正常点
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=f'{details["label"]} (Normal)',
                markerfacecolor=details['color'],
                markersize=8,
                markeredgecolor='w',
                alpha=0.6
            )
        )
        # 更换信号点
        legend_elements.append(
            plt.Line2D(
                [0], [0],
                marker='X',
                color='w',
                label=f'{details["label"]} (Alert)',
                markerfacecolor=details['color'],
                markersize=10,
                markeredgecolor='black',
                linewidth=1.2,
                alpha=0.9
            )
        )

    # 添加图例
    ax1.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), fontsize=10)

    # 调整布局以防止图例被裁剪
    plt.tight_layout()

    # 保存图像
    if result_path is not None:
        plt.savefig(result_path, dpi=150, bbox_inches='tight')
    plt.close()


def filter_values(angle_list, torque_list, mode=0, min_value=0):
    """
    同步过滤angle和torque中的负值数据（保留两者均≥min_value的点）
    返回过滤后的(angle_sublist, torque_sublist)
    """
    # 生成索引列表，仅保留两者均≥min_value的位置 
    if mode == 0:
        valid_indices = [idx for idx, (a, t) in enumerate(zip(angle_list, torque_list)) if a > min_value and t > min_value] 
    elif mode == 1:
        valid_indices = [idx for idx, (a, t) in enumerate(zip(angle_list, torque_list)) if a < min_value and t > min_value] 
    elif mode == 2:
        valid_indices = [idx for idx, (a, t) in enumerate(zip(angle_list, torque_list)) if t > min_value] 
    else:
        return angle_list, torque_list
    # 提取过滤后的子列表（若没有有效数据返回空列表）
    angle_sub = [angle_list[idx] for idx in valid_indices] if valid_indices else []
    torque_sub = [torque_list[idx] for idx in valid_indices] if valid_indices else []
    return angle_sub, torque_sub


# ==================== 绘图函数 ====================
def plot_comparison_test(row, random_state=2):

    # 创建画布（总尺寸根据子图数量调整）
    fig, axes = plt.subplots(1, 3, figsize=(24, 5), constrained_layout=True)
    axes = axes.ravel()  # 展平为一维数组方便循环
    raw_angle = row['angledata']
    raw_torque = row['torquedata']

    # 1. 
    filtered_angle, filtered_torque = filter_values(raw_angle, raw_torque, min_value=0)

        # 绘制曲线（使用固定颜色区分螺丝索引，透明度区分样本）, color=screw_color_map[s_idx]
        ax.plot(filtered_angle, filtered_torque, alpha=1 - (sample_idx * 0.02), linewidth=1.2, marker='', linestyle='-', label=row['hsgsn'])
    # 添加图例（仅显示当前子图的样本）
    for ax in axes
    ax.legend(loc='best', fontsize=8)    # frameon=True, 
    ax.grid(True, linestyle='--', alpha=0.4)
    axes.set_xlabel('Angle', fontsize=10)
    axes[0].set_ylabel('Torque', fontsize=10)
    # 添加总标题
    fig.suptitle('Torque vs Angle Comparison by Recommended Action', fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()