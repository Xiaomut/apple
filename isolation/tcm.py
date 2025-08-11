import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import time
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.seasonal import seasonal_decompose
from collections import defaultdict
import shap
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


class ScrewdriverReplacementAnalyzer:
    def __init__(self, window_size=20, failure_threshold=3, screw_positions=[1, 2, 4, 41], top_causes=3, anomaly_num=-1.5, contamination=0.02):
        """
        初始化螺丝刀更换分析器
        
        参数:
        - window_size: 滑动窗口大小
        - failure_threshold: 连续异常次数阈值
        - screw_positions: 螺丝位置顺序列表（按拧紧顺序）
        - top_causes: 输出前N个主要原因
        """
        self.window_size = window_size
        self.failure_threshold = failure_threshold
        self.screw_positions = screw_positions
        self.top_causes = top_causes
        self.anomaly_num = anomaly_num
        self.contamination = contamination
        # 特征权重（根据工程经验调整）
        self.feature_weights = {
            'torque_max': 1.0,          # 最高权重：最大扭矩直接反映拧紧力度
            'clamp_torque': 0.9,         # 夹紧扭矩不足是主要失效模式
            'angle_max': 0.85,           # 角度超差影响装配精度
            'clamp_angle': 0.8,         # 
            'Normalized_integral': 0.8,  # 能量积分反映整体拧紧质量
            'gradient_mean': 0.7,        # 扭矩变化率异常预示摩擦增大
            'FFT_Max': 0.6               # 高频振动异常
        }
        
        # 位置权重（根据拧紧顺序重要性调整）
        self.position_weights = {1: 0.1, 2: 0.3, 4: 0.3, 41: 0.3}
        # 中间结果存储
        self.position_dfs = {}          # 各位置预处理后数据
        self.position_results = {}      # 各位置分析结果（含贡献度）
        self.scalers = {}               # 各位置标准化器（备用）
        self.combined_df = None         # 合并后总结果
        self.cause_records = []         # 更换原因记录
        self.shap_values_cache = {}     # SHAP值缓存（加速计算）

    def _preprocess_position_data(self, pos_df, pos):
        """预处理单个螺丝位置数据"""
        # 关键特征列表（按重要性排序）
        features = list(self.feature_weights.keys())
        existing_features = [f for f in features if f in pos_df.columns]
        
        # 缺失值处理（后向填充+中位数填充）
        pos_df = pos_df.copy()
        for f in existing_features:
            if pos_df[f].isna().any():
                pos_df[f] = pos_df[f].ffill().bfill()
                # 长期缺失用中位数（超过窗口大小10倍的缺失视为异常）
                if pos_df[f].isna().sum() > 0.1 * self.window_size:
                    pos_df[f] = pos_df[f].fillna(pos_df[f].median())
        self.scalers[pos] = StandardScaler()
        pos_df[existing_features] = self.scalers[pos].fit_transform(pos_df[existing_features])
        
        # 滑动窗口标准化（时间序列感知）
        for f in existing_features:
            rolling = pos_df[f].rolling(window=self.window_size, min_periods=1)
            pos_df[f'{f}_zscore'] = (pos_df[f] - rolling.mean()) / (rolling.std() + 1e-8)

        pos_df['direct_failure'] = (pos_df['Pass/Fail'] != 'Pass').astype(int)
        # 记录位置信息
        pos_df['screw_position'] = pos
        
        return pos_df

    def _calculate_feature_contributions(self, pos_df):
        """计算各特征对异常的贡献度"""
        # 异常检测分数（Isolation Forest）
        clf = IsolationForest(contamination=self.contamination, random_state=42)
        
        X = pos_df[[f'{f}_zscore' for f in self.feature_weights.keys()]].bfill().ffill()

        # 计算加权异常分数
        y_pred = clf.fit_predict(X)
        anomaly_scores = -clf.score_samples(X)  # 转换为"越异常分数越高"
        pos_df['anomaly_score'] = anomaly_scores
        pos_df['is_anomaly'] = (y_pred == -1).astype(int)

        # 特征贡献度计算（基于z-score偏离程度）
        contribution = defaultdict(float)
        for f in self.feature_weights.keys():
            z_col = f'{f}_zscore'
            if z_col in pos_df.columns:
                # 异常时贡献度=权重*|z-score|，正常时贡献度=0
                contribution[f] = self.feature_weights[f] * pos_df[z_col].fillna(0).abs()
        
        # 归一化总贡献度
        total_contribution = pos_df[list(contribution.keys())].sum(axis=1) + 1e-8
        for f in contribution:
            contribution[f] = contribution[f] / total_contribution
        
        # 记录直接失败贡献
        contribution['direct_failure'] = pos_df['direct_failure'] * 0.5  # 固定权重
        
        return pos_df, contribution

    def analyze_position(self, pos_df, pos):
        """分析单个螺丝位置数据"""
        start_time = time.time()
        # 预处理
        processed_df = self._preprocess_position_data(pos_df, pos)
        self.position_dfs[pos] = processed_df
        # 计算特征贡献
        analyzed_df, contributions = self._calculate_feature_contributions(processed_df)
        self.position_results[pos] = analyzed_df
        
        # 记录更换原因（仅当检测到异常时）
        anomaly_mask = analyzed_df['is_anomaly'] == 1
        if anomaly_mask.any():
            for idx in analyzed_df.index[anomaly_mask]:
                # 获取前N大贡献特征
                top_features = sorted(contributions.items(), key=lambda x: x[1][idx], reverse=True)[:self.top_causes]
                
                # 构造原因记录
                cause_record = {
                    'time': analyzed_df.loc[idx, 'DateTime'],
                    'filename': analyzed_df.loc[idx, 'filename'],
                    'screw_position': pos,
                    'anomaly_score': analyzed_df.loc[idx, 'anomaly_score'],
                    'main_cause': top_features[0][0],
                    'top_causes': [(f[0], round(f[1][idx], 4)) for f in top_features],
                    # 'failure_count': analyzed_df.loc[idx, 'failure_count']  # 后续合并时填充
                }
                self.cause_records.append(cause_record)
        
        print(f"处理位置 {pos} 完成，耗时 {time.time()-start_time:.2f}s")
        return analyzed_df

    def _combine_results(self):
        """合并各位置分析结果"""
        # 按原始顺序合并按时间排序
        combined_df = pd.concat([self.position_results[pos] for pos in self.screw_positions]).sort_values(['DateTime', 'filename'])
        
        # 计算加权异常分数
        combined_df['weighted_anomaly'] = 0
        for pos in self.screw_positions:
            mask = combined_df['screw_position'] == pos
            combined_df.loc[mask, 'weighted_anomaly'] = pd.to_numeric(combined_df.loc[mask, 'anomaly_score'], errors='coerce') * self.position_weights[pos]

        # 标记连续失败
        combined_df['failure_count'] = combined_df['is_anomaly'].rolling(window=self.window_size*len(self.screw_positions), min_periods=1).sum()

        # 生成更换信号
        combined_df['replacement_signal'] = (combined_df['failure_count'] >= self.failure_threshold).astype(bool)
        combined_df['first_replacement'] = combined_df.groupby(['SN'])['replacement_signal'].transform(lambda x: x & ~x.shift(1).fillna(False)).astype(int)
        
        return combined_df

    def analyze_all_positions(self, df):
        """分析所有螺丝位置数据"""
        # 按位置分组处理
        for pos in self.screw_positions:
            pos_df = df[df['screw position'] == pos].copy()
            if not pos_df.empty:
                pos_df = self.analyze_position(pos_df, pos)

        self.combined_df = self._combine_results()
        
        return self.combined_df

    def plot_analysis_results(self, output_path: str = "replacement_analysis.png", figsize: tuple = (16, 12), show_top_causes: bool = True) -> None:
        """绘制分析结果可视化图（性能优化 & 可读性增强版）"""
        if self.combined_df is None or self.combined_df.empty:
            print("警告：没有可绘制的数据（combined_df 为空）")
            return

        self.combined_df['DateTime'] = pd.to_datetime(self.combined_df['DateTime'])
        # === 1. 高效降采样（避免卡顿）===
        # 如果数据太多，按时间降采样（例如每分钟一个点）
        df_plot = self.combined_df.sort_values('DateTime').copy()
        if len(df_plot) > 500:
            df_plot = df_plot.set_index('DateTime').resample('1min').first().reset_index()
            df_plot = df_plot.ffill().bfill()  # 填充缺失
        else:
            df_plot = df_plot.iloc[::max(1, len(df_plot)//500)]  # 简单降采样

        # === 2. 准备更换事件点 ===
        replacement_mask = df_plot['replacement_signal'] == 1
        first_replace_mask = df_plot['first_replacement'] == 1

        # 提取事件点（避免空 DataFrame 报错）
        replacements = df_plot[replacement_mask]
        first_replacements = df_plot[first_replace_mask]

        # === 3. 创建子图（优化布局）===
        fig, axes = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        ax1, ax2, ax3, ax4 = axes.flat

        # === 图1: 主时间序列 - 加权异常分数与更换信号 ===
        ax1.plot(df_plot['DateTime'], df_plot['weighted_anomaly'], color='blue', alpha=0.6, linewidth=1, label='加权异常分数')
        
        if not replacements.empty:
            ax1.scatter(replacements['DateTime'], replacements['weighted_anomaly'], color='red', s=30, alpha=0.8, label='更换触发点', edgecolors='white', linewidth=0.5)

        # if not first_replacements.empty:
        #     ax1.scatter(first_replacements['DateTime'], first_replacements['weighted_anomaly'], color='purple', s=60, marker='*', label='首次更换', zorder=5)

        ax1.set_title('螺丝拧紧异常分数与更换信号', fontsize=12, fontweight='bold')
        ax1.set_ylabel('加权异常分数', fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # --- 优化横坐标显示 ---
        self._format_datetime_axis(ax1)

        # === 图2: 异常主因分布（柱状图替代饼图，更清晰）===
        if self.cause_records:
            cause_counts = pd.Series([record['main_cause'] for record in self.cause_records]).value_counts(normalize=True)
            sns.barplot(x=cause_counts.index, y=cause_counts.values, ax=ax2, palette='viridis')
            ax2.set_title('异常主原因分布')
            ax2.set_ylabel('占比')
            ax2.tick_params(axis='x', rotation=30)
        else:
            ax2.text(0.5, 0.5, '无异常记录', ha='center', va='center')
            ax2.set_title('异常主原因分布')

        # === 图3: 关键特征趋势（选择前2个位置 + 关键特征）===
        if show_top_causes and self.position_results:
            selected_pos = self.screw_positions[:2]
            colors = plt.get_cmap('tab10')(np.linspace(0, 1, len(selected_pos) * 2))
            color_idx = 0

            for pos in selected_pos:
                if pos not in self.position_results:
                    continue
                pos_df = self.position_results[pos].sort_values('DateTime')
                # 降采样
                pos_df_sample = pos_df.iloc[::max(1, len(pos_df)//200)]

                for feature in ['torque_max_zscore', 'clamp_torque_zscore']:
                    if feature in pos_df_sample.columns:
                        ax3.plot(pos_df_sample['DateTime'], pos_df_sample[feature],
                                label=f'位置{pos} {feature.replace("_zscore", "")}',
                                alpha=0.7, linewidth=1.2, color=colors[color_idx])
                        color_idx += 1

            ax3.axhline(y=self.anomaly_num, color='red', linestyle='--', alpha=0.6, linewidth=1, label='异常阈值')
            ax3.set_title('关键特征标准化趋势', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Z-Score', fontsize=10)
            ax3.legend(fontsize=8, loc='upper right')
            ax3.grid(True, alpha=0.3)
            self._format_datetime_axis(ax3)
        else:
            ax3.text(0.5, 0.5, '无特征数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('关键特征趋势', fontsize=12, fontweight='bold')

        # === 图4: 位置权重分布（柱状图）===
        pos_weights = pd.Series(self.position_weights)
        sns.barplot(x=pos_weights.index, y=pos_weights.values, ax=ax4, palette='rocket')
        ax4.set_title('螺丝位置权重分布')
        ax4.set_ylabel('权重值')
        ax4.tick_params(axis='x', rotation=30)

        # === 保存图像（高清）===
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)  # 防止内存泄漏
        print(f"分析结果图已保存至 {output_path}")

    def _format_datetime_axis(self, ax):
        """统一优化时间坐标轴显示"""
        # 自动选择合适的日期格式
        locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
        formatter = mdates.ConciseDateFormatter(locator)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.tick_params(axis='x', rotation=30, labelsize=9)

def plot_time_series_decomposition(combined_df, pos=4):
    """绘制时间序列分解图"""
    if combined_df is None:
        return
    
    # 获取特定位置的数据
    pos_data = combined_df[combined_df['screw position'] == pos].copy()
    
    # 准备数据
    datetime = pos_data['DateTime']
    torque = pos_data['torque_max']
    angle = pos_data['angle_max']
    
    # 创建季节性分解
    torque_result = seasonal_decompose(torque, model='additive', period=24 * 60)  # 假设每天的周期
    angle_result = seasonal_decompose(angle, model='additive', period=24 * 60)
    
    # 创建画布
    fig, axes = plt.subplots(2, 2, figsize=(20, 15))
    
    # 扭矩分解
    axes[0, 0].plot(datetime, torque, label='Original')
    axes[0, 0].plot(datetime, torque_result.trend, label='Trend')
    axes[0, 0].plot(datetime, torque_result.seasonal, label='Seasonality')
    axes[0, 0].plot(datetime, torque_result.resid, label='Residuals')
    axes[0, 0].set_title('Torque Decomposition')
    axes[0, 0].legend()
    
    # 角度分解
    axes[0, 1].plot(datetime, angle, label='Original')
    axes[0, 1].plot(datetime, angle_result.trend, label='Trend')
    axes[0, 1].plot(datetime, angle_result.seasonal, label='Seasonality')
    axes[0, 1].plot(datetime, angle_result.resid, label='Residuals')
    axes[0, 1].set_title('Angle Decomposition')
    axes[0, 1].legend()
    
    # 绘制分解组件的统计信息
    stats = pd.DataFrame({
        'Torque_Trend': torque_result.trend.describe(),
        'Torque_Seasonal': torque_result.seasonal.describe(),
        'Torque_Residual': torque_result.resid.describe(),
        'Angle_Trend': angle_result.trend.describe(),
        'Angle_Seasonal': angle_result.seasonal.describe(),
        'Angle_Residual': angle_result.resid.describe()
    })
    
    # 异常分数分解
    axes[1, 0].plot(datetime, pos_data['anomaly_score'], label='Original')
    axes[1, 0].set_title('Anomaly Score Over Time')
    axes[1, 0].legend()
    
    # 异常频率随时间变化
    df_by_hour = pos_data.groupby(pd.to_datetime(pos_data['DateTime']).dt.hour)['anomaly'].mean().reset_index()
    axes[1, 1].plot(df_by_hour['DateTime'], df_by_hour['anomaly'], marker='o')
    axes[1, 1].set_title('Average Anomaly Score by Hour')
    axes[1, 1].set_xticks(range(24))
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(r"datas\NPI数据收集\LG-1\tcmdata.csv", encoding='utf8', on_bad_lines='skip')
    df['filename'] = df['filename'].replace(r'_pos\d+', '', regex=True)
    df = df.sort_values(['DateTime', 'filename'])  #

    # fail_cols = [x for x in df.columns if 'failure_' in x or '_result' in x]
    # for col in fail_cols:
    #     if df[col].value_counts().to_dict().get('Fail', 0) > 20:
    #         print(f"{col}: {df[col].value_counts().to_dict()}")

    df_use = df.copy()
    df_use.replace("Not_Analyzed", None, inplace=True)

    features = [
        'angle_start', 'angle_max', 'torque_start', 'torque_max', 'torque_final',
        'clamp_torque', 'clamp_angle', 'torque_sit', 'angle_sit',
        'max_torque_before_sit', 'angle_at_maxTorque_before_sit',
        'angular_work_total', 'angular_work_after_sit',
        'angular_work_before_sit', 'clamp_torque_percentage',
        'torque_sit_percentage', 'Normalized_integral', 'gradient_mean',
        'gradient_stdev', 'max derivative', 'max double derivative',
        'gradient_negative_mean', 'gradient_negative_stdev', 'skew_total',
        'skew_negative', 'kurtosis_total', 'kurtosis_negative', 'FFT_Max',
        'FFT_Freq_360', 'FFT_Average', 'FFT_Median', 'FFT_StDev',
        'LSRL_int', 'LSRL_r^2', 'LSRL_slope_norm',
        'LSRL_int_norm', 'LSRL_r^2_norm'
    ]
    for col in features:
        df_use[col] = df_use[col].astype(float)

    screw_positions = [4,1,2,41]
    # 初始化分析器
    analyzer = ScrewdriverReplacementAnalyzer(
        window_size=50,
        failure_threshold=3,
        screw_positions=screw_positions,
        top_causes=3
    )

    # 执行分析
    print("Starting analysis...")
    combined_df = analyzer.analyze_all_positions(df_use)
    analyzer.plot_analysis_results(output_path=r"datas\imgs\looking\res.png", figsize=(18, 14))
    # plot_time_series_decomposition(combined_df)

    # 示例代码：详细查看异常点
    low_anomaly_replacements = combined_df[(combined_df['replacement_signal'] == 1) & (combined_df['weighted_anomaly'] < 0.1)]
    print(low_anomaly_replacements[['DateTime', 'weighted_anomaly', 'torque_max_zscore', 'clamp_torque_zscore']])
    print()
