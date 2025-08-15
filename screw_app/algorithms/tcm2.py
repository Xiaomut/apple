import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import shap
import pickle
from collections import defaultdict, deque
import time
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
from screw_app.algorithms.screw_anlysis import plot_screw_analysis_trend

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ScrewdriverReplacementAnalyzer:
    """
    螺丝刀更换智能监测分析器（优化版）
    
    功能：
    - 多维度异常检测（统计规则 + 无监督模型）
    - 基于 SHAP 的可解释性贡献分析
    - 连续劣化趋势判断
    - 支持多位置、多特征加权融合
    """

    def __init__(self, 
                 window_size: int = 20,
                 failure_threshold: int = 3,
                 screw_positions: List[int] = [1, 2, 4, 41],
                 top_causes: int = 3,
                 contamination: float = 0.02,
                 rule_threshold_z: float = 3.0,
                 min_anomaly_duration: int = 2):
        """
        初始化参数
        
        参数:
        - window_size: 滑动窗口大小（用于 z-score 和趋势分析）
        - failure_threshold: 触发更换的连续异常次数阈值
        - screw_positions: 螺丝位置列表
        - top_causes: 输出前 N 个主要原因
        - contamination: Isolation Forest 异常比例估计
        - rule_threshold_z: 规则法异常判定的 z-score 阈值（默认 ±3σ）
        - min_anomaly_duration: 连续异常最小持续长度（防抖）
        """
        self.window_size = window_size
        self.failure_threshold = failure_threshold
        self.screw_positions = screw_positions
        self.top_causes = top_causes
        self.contamination = contamination
        self.rule_threshold_z = rule_threshold_z
        self.min_anomaly_duration = min_anomaly_duration

        # 特征权重（可后续通过反馈学习调整）
        self.feature_weights = {
            'torque_max': 1.0,          # 最高权重：最大扭矩直接反映拧紧力度
            'torque_final': 1.2,          # 
            # 'clamp_angle': 0.7,         # 夹紧扭矩不足是主要失效模式
            'gradient_negative_mean': 0.7,         # 
            'angle_max': 0.8,           # 角度超差影响装配精度
            # 'angle_sit': 0.6,           # 
            'angular_work_total': 0.9,  # 能量积分反映整体拧紧质量
            'gradient_stdev': 0.9,        # 扭矩变化率异常预示摩擦增大
            'FFT_Max': 0.9               # 高频振动异常
        }

        # 位置权重（关键位置更高）
        self.position_weights = {1: 0.1, 2: 0.3, 4: 0.3, 41: 0.3}

        # 存储结构
        self.position_dfs = {}
        self.position_results = {} 
        self.scalers = {}
        self.combined_df = None
        self.cause_records = []
        self.shap_values_cache = {}
        self.models = {}  # 存储每个位置的解释模型

        logger.info(f"Analyzer initialized with window={window_size}, threshold={failure_threshold}")

    def _preprocess_position_data(self, pos_df: pd.DataFrame, pos: int) -> pd.DataFrame:
        """预处理单个位置数据"""
        pos_df = pos_df.copy().reset_index(drop=True)
        features = list(self.feature_weights.keys())
        existing_features = [f for f in features if f in pos_df.columns]

        if len(existing_features) == 0:
            logger.warning(f"Position {pos}: No valid features found.")
            return pos_df

        # 缺失值处理
        for f in existing_features:
            if pos_df[f].isna().any():
                # 前向+后向填充
                pos_df[f] = pos_df[f].ffill().bfill()
                if pos_df[f].isna().any():
                    pos_df[f] = pos_df[f].fillna(pos_df[f].median())
                    logger.debug(f"Position {pos}, feature {f}: filled with median {pos_df[f].median()}")

        # 滑动 z-score（时间序列感知标准化）
        for f in existing_features:
            roll = pos_df[f].rolling(window=self.window_size, min_periods=1)
            # 不要进行归一化，因为归一化后容易把趋势抹平
            pos_df[f'{f}_zscore'] = (pos_df[f] - roll.mean())   # / roll.std().replace(0, 1e-8)
            # pos_df[f'{f}_zscore'] = pos_df[f]   # (pos_df[f] - roll.mean()) / roll.std().replace(0, 1e-8)

        # 标记直接失败（Pass/Fail 列）
        pos_df['direct_failure'] = (pos_df['Pass/Fail'] != 'Pass').astype(int)

        pos_df['screw_position'] = pos
        pos_df['DateTime'] = pd.to_datetime(pos_df['DateTime'], errors='coerce')
        return pos_df

    def _detect_anomaly_hybrid(self, pos_df: pd.DataFrame) -> pd.Series:
        """混合异常检测：规则 + 模型"""
        features = [f'{f}_zscore' for f in self.feature_weights.keys()]
        X = pos_df[features].fillna(0)

        # # 方法1：规则法（z-score 超阈值）
        # rule_anomaly = (X.abs() > self.rule_threshold_z).any(axis=1).astype(int)

        # 方法2：Isolation Forest
        clf = IsolationForest(contamination=self.contamination, random_state=42, n_estimators=100)
        model_anomaly = clf.fit_predict(X)
        anomaly_scores = -clf.score_samples(X)  # 转换为"越异常分数越高"
        pos_df['anomaly_score'] = anomaly_scores
        model_anomaly = (model_anomaly == -1).astype(int)
        # # 融合策略：任一方法判定为异常即为异常  
        # hybrid_anomaly = (rule_anomaly | model_anomaly).astype(int)
        pos_df['is_anomaly'] = model_anomaly
        return pos_df

    def _calculate_shap_contributions(self, pos_df: pd.DataFrame, pos: int) -> Tuple[pd.DataFrame, Dict]:
        """使用 SHAP 解释模型计算特征贡献度"""
        features = [f'{f}_zscore' for f in self.feature_weights.keys()]
        X = pos_df[features].fillna(0)
        y = pos_df['is_anomaly'].clip(0, 1)  # 确保为 0/1

        if y.nunique() < 2:
            logger.warning(f"Position {pos}: Not enough anomaly variation to train SHAP model.")
            contributions = {f: pd.Series(np.zeros(len(pos_df)), index=pos_df.index) for f in self.feature_weights}
            return pos_df, contributions

        try:
            # 使用轻量级模型解释
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
            model.fit(X, y)
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X).values

            # 取正类（anomaly=1）的 SHAP 值（绝对值贡献）
            if len(shap_values.shape) == 3:
                shap_vals = np.abs(shap_values[:, :, 1])  # 多分类取 anomaly 类
            else:
                shap_vals = np.abs(shap_values)

            contributions = {}
            for i, f in enumerate(features):
                raw_feature = f.replace('_zscore', '')
                contributions[raw_feature] = pd.Series(shap_vals[:, i], index=pos_df.index)

            # 归一化贡献（按行）
            contrib_df = pd.DataFrame(contributions)
            row_sums = contrib_df.sum(axis=1) + 1e-8
            for col in contrib_df.columns:
                contributions[col] = contrib_df[col] / row_sums

            self.models[pos] = model  # 缓存模型
            logger.debug(f"SHAP model trained for position {pos}")

        except Exception as e:
            logger.error(f"SHAP calculation failed for pos {pos}: {e}")
            # 回退到简单 z-score 方法
            contributions = {}
            for f in self.feature_weights.keys():
                z_col = f'{f}_zscore'
                if z_col in pos_df.columns:
                    contrib = pos_df[z_col].abs() * self.feature_weights[f]
                else:
                    contrib = pd.Series(np.zeros(len(pos_df)), index=pos_df.index)
                contributions[f] = contrib
            # 归一化
            temp_df = pd.DataFrame(contributions)
            row_sums = temp_df.sum(axis=1) + 1e-8
            for col in temp_df.columns:
                contributions[col] = temp_df[col] / row_sums

        return pos_df, contributions

    def analyze_position(self, pos_df: pd.DataFrame, pos: int) -> pd.DataFrame:
        """分析单个螺丝位置"""
        start_time = time.time()
        logger.info(f"Starting analysis for position {pos} with {len(pos_df)} records.")

        # 1. 预处理
        processed_df = self._preprocess_position_data(pos_df, pos)
        if len(processed_df) == 0:
            logger.warning(f"Position {pos}: No data after preprocessing.")
            return processed_df

        # 2. 混合异常检测
        processed_df = self._detect_anomaly_hybrid(processed_df)

        # # 3. 连续异常计数（仅连续异常才累计）
        # consecutive_count = 0
        # for i in range(len(processed_df)):
        #     if processed_df.iloc[i]['is_anomaly_rule'] == 1:
        #         consecutive_count += 1
        #     else:
        #         consecutive_count = 0
        #     if consecutive_count >= self.min_anomaly_duration:
        #         processed_df.iloc[i, processed_df.columns.get_loc('is_anomaly')] = 1

        # 4. 计算 SHAP 贡献度
        _, contributions = self._calculate_shap_contributions(processed_df, pos)

        # 5. 计算加权异常得分
        processed_df['anomaly_score'] = processed_df['anomaly_score'] * processed_df[[f'{f}_zscore' for f in self.feature_weights.keys()]].abs().mean(axis=1, skipna=True)

        # 6. 连续异常信号（滑动窗口内累计异常次数）
        processed_df['failure_count'] = processed_df['is_anomaly'].rolling(window=self.window_size, min_periods=1).sum()
        processed_df['replacement_signal'] = (processed_df['failure_count'] >= self.failure_threshold).astype(int)

        # 7. 记录更换原因
        replacement_mask = processed_df['replacement_signal'] == 1
        if replacement_mask.any():
            for idx in processed_df.index[replacement_mask]:
                top_features = sorted(
                    [(f, contributions[f][idx]) for f in contributions],
                    key=lambda x: x[1], reverse=True
                )[:self.top_causes]

                cause = {
                    'time': processed_df.loc[idx, 'DateTime'],
                    'filename': processed_df.loc[idx, 'filename'],
                    'screw_position': pos,
                    'sn': processed_df.loc[idx, 'SN'],
                    'screw_number': processed_df.loc[idx, 'screw number'],
                    'main_cause': top_features[0][0],
                    'top_causes': [(f[0], round(f[1], 4)) for f in top_features],
                    'anomaly_score': round(processed_df.loc[idx, 'anomaly_score'], 4),
                    'failure_count': int(processed_df.loc[idx, 'failure_count'])
                }
                self.cause_records.append(cause)

        self.position_dfs[pos] = processed_df
        self.position_results[pos] = processed_df.copy()
        logger.info(f"Position {pos} processed in {time.time() - start_time:.2f}s")
        return processed_df

    def analyze_all_positions(self, df: pd.DataFrame) -> pd.DataFrame:
        """分析所有位置"""
        if df.empty:
            logger.error("Input DataFrame is empty.")
            return pd.DataFrame()

        # 确保必要字段存在
        required_cols = ['screw position', 'DateTime', 'SN', 'filename']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return pd.DataFrame()

        # 按位置分组分析
        for pos in self.screw_positions:
            pos_df = df[df['screw position'] == pos].copy()
            if not pos_df.empty:
                self.analyze_position(pos_df, pos)
            else:
                logger.warning(f"No data for position {pos}")

        if not self.position_dfs:
            logger.error("No data processed for any position.")
            return pd.DataFrame()

        # 合并所有位置结果
        combined_list = []
        for pos in self.screw_positions:
            if pos in self.position_results:
                subset = self.position_results[pos][[
                    'DateTime', 'filename', 'SN', 'screw position', 'screw number',
                    'torque_max', 'clamp_torque', 'angle_max', 'is_anomaly',
                    'anomaly_score', 'failure_count', 'replacement_signal'
                ]].copy()
                # 加权异常得分
                weight = self.position_weights.get(pos, 1.0)
                subset['weighted_anomaly'] = subset['anomaly_score'] * weight
                combined_list.append(subset)

        if not combined_list:
            return pd.DataFrame()

        self.combined_df = pd.concat(combined_list).sort_values(['DateTime', 'filename']).reset_index(drop=True)
        logger.info(f"Analysis completed. Total records: {len(self.combined_df)}")
        logger.info(f"Total replacement signals triggered: {self.combined_df['replacement_signal'].sum()}")

        return self.combined_df

    def get_top_causes_summary(self) -> pd.DataFrame:
        """获取更换原因汇总"""
        if not self.cause_records:
            return pd.DataFrame()
        summary_df = pd.DataFrame(self.cause_records)
        return summary_df

    def reset(self):
        """重置分析器状态"""
        self.position_dfs.clear()
        self.position_results.clear()
        self.cause_records.clear()
        self.combined_df = None
        self.models.clear()
        logger.info("Analyzer state reset.")


def train(path, window_size=15, failure_threshold=4, top_causes=3, contamination=0.05, model_path=None, cause_path=None):
    df = pd.read_csv(path, encoding='utf8', on_bad_lines='skip')
    df['filename'] = df['filename'].replace(r'_pos\d+', '', regex=True)
    df = df.sort_values(['DateTime', 'filename'])  #

    df_use = df[df['DateTime'] > '2025/06/05']     # .copy()
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

    # 初始化分析器
    analyzer = ScrewdriverReplacementAnalyzer(
        window_size=window_size,
        failure_threshold=failure_threshold,
        screw_positions=[4,1,2,41],
        top_causes=top_causes,
        contamination=contamination
    )
    result_df = analyzer.analyze_all_positions(df)

    # 获取更换原因
    cause_df = analyzer.get_top_causes_summary()
    if cause_path is not None:
        cause_df.to_csv(cause_path, index=False)
    print(cause_df.head())

    analyzer.shap_values_cache = {}  # 避免 pickle 报错

    with open(model_path, 'wb') as f:
        pickle.dump(analyzer, f)


def valid(model_path='isolation/pth/screwdriver_analyzer_trained.pkl', fig_path=r"datas\imgs\looking\new.png"):
    with open(model_path, 'rb') as f:
        analyzer = pickle.load(f)

    plot_screw_analysis_trend(analyzer.combined_df, fig_path)


if __name__ == "__main__":

    data_type = 'SM'
    model_path = f"isolation/pth/screwdriver_analyzer_{data_type}_trained.pkl"

    train(path=f"datas/NPI数据收集/{data_type}-1/tcmdata.csv", window_size=20, failure_threshold=4, top_causes=3, contamination=0.02, model_path=model_path, cause_path=f"screw_app/static/results/cause_{data_type}.csv")
    # exit(0)
    valid(model_path, fig_path=f"datas/imgs/looking/{data_type}_res.png")

