from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题


screw_order = [1, 2, 4, 41]


def validate_data(row):
    """验证每个样本的angle和torque数据长度是否一致"""
    if len(row['angledata']) != len(row['torquedata']):
        print(f"警告：样本 {row.name} 的angledata和torquedata长度不一致（{len(row['angledata'])} vs {len(row['torquedata'])}），已跳过")
        return False
    return True


def filter_negative_values(angle_list, torque_list, min_value=0):
    """
    同步过滤angle和torque中的负值数据（保留两者均≥min_value的点）
    返回过滤后的(angle_sublist, torque_sublist)
    """
    # 生成索引列表，仅保留两者均≥min_value的位置 
    valid_indices = [
        idx for idx, (a, t) in enumerate(zip(angle_list, torque_list)) 
        if a > min_value and t > min_value
    ]
    # 提取过滤后的子列表（若没有有效数据返回空列表）
    angle_sub = [angle_list[idx] for idx in valid_indices] if valid_indices else []
    torque_sub = [torque_list[idx] for idx in valid_indices] if valid_indices else []
    return angle_sub, torque_sub


def getValidGroups(df_use):
    valid_df = df_use[df_use.apply(validate_data, axis=1)]
    grouped = valid_df.groupby(['recommended_action', 'screwindex'])

    # ==================== 绘图准备 ====================
    valid_groups = [
        (action, s_idx) 
        for action in ['Continue', 'Rework', 'FA']  # 按推荐动作固定顺序
        for s_idx in screw_order  # 按自定义螺丝索引顺序
        if (action, s_idx) in grouped.groups
    ]
    return valid_groups, grouped


# -----------------------------
# 3. 提取每个螺丝打螺丝过程的特征
# -----------------------------
def extract_features(row):
    raw_angle = row['angledata']
    raw_torque = row['torquedata']
    angle_v, torque_v = filter_negative_values(raw_angle, raw_torque, min_value=0)

    if len(angle_v) < 2:
        return pd.Series([np.nan]*12, index=['hsgsn', 'time', 'max_torque','angle_at_max','auc', 'slope_mid','slope_rise_late','total_angle', 'std_torque','row_span','valid', 'recommended_action'])

    # 1. 最大扭力 & 对应角度
    max_torque = np.max(torque_v)
    angle_at_max = angle_v[np.argmax(torque_v)]

    # 2. ✅ AUC：以 angle 为横坐标的积分 ∫ torque d(angle)
    auc = np.trapz(torque_v, angle_v)  # 单位: N·m·°

    # 斜率
    mid_angle = angle_v[len(angle_v) // 2]
    mid_torque = torque_v[len(angle_v) // 2]
    slope_mid = (max_torque - mid_torque) / (np.max(angle_v) - mid_angle + 1e-8)
    
    # 上升段斜率：前50%上升部分
    peak_idx = np.argmax(torque_v)
    mid_torque = 0.5 * max_torque
    if peak_idx < 2:
        slope_rise_late = 0.0
    else:
        # 找出后半上升段的索引
        late_rise_mask = (angle_v >= mid_torque) & (torque_v <= max_torque)
        late_rise_points = np.where(late_rise_mask)[0]

        if len(late_rise_points) >= 2:
            idx_start = late_rise_points[0]
            idx_end = late_rise_points[-1]
            delta_angle = angle_v[idx_end] - angle_v[idx_start]
            delta_torque = torque_v[idx_end] - torque_v[idx_start]
            slope_rise_late = delta_torque / (delta_angle + 1e-8)  # 防除零
        else:
            slope_rise_late = 0.0
    
    # 4. 总角度跨度（有效段）
    total_angle = angle_v[-1] - angle_v[0]
    # 5. 扭力标准差
    std_torque = np.std(torque_v)
    # 7. row_in_file 跨度
    row_span = len(angle_v)

    return pd.Series([
        row['hsgsn'], row['time'], max_torque, angle_at_max, auc, slope_mid, slope_rise_late, total_angle, std_torque, row_span, True, row['recommended_action']
    ], index=[
        'hsgsn', 'time', 'max_torque','angle_at_max','auc', 'slope_mid','slope_rise_late','total_angle', 'std_torque','row_span','valid', 'recommended_action'
    ])


def process_location_data(df_use: pd.DataFrame):
    loc_groups = df_use['screwindex'].unique()
    all_features_dfs = []

    for loc in loc_groups:
        group_df = df_use[df_use['screwindex'] == loc].copy().sort_values('time')
        print(f"\n🔧 正在处理 loc={loc}，共 {len(group_df)} 个螺丝")
        
        # 按 screw_id 分组提取特征
        features = group_df.apply(extract_features, axis=1)
        
        # 检查并处理 NaN 值
        if features.isna().any(axis=None):  # 如果存在任何 NaN 值
            features['is_abnormal'] = features.isna().any(axis=1)  # 标记包含 NaN 的行为 True
            features['abnormal_reason'] = np.where(features['is_abnormal'], '批杆问题(可能)', '')  # 假设 NaN 是由批杆问题引起的
            
            # 对于有 NaN 值的行，你可以选择填充、删除或进一步处理
            # 这里我们选择用默认值填充以便继续处理
            features.fillna({'max_torque': -1, 'angle_at_max': -1, 'auc': -1,
                             'slope_rise_late': -1, 'total_angle': -1,
                             'std_torque': -1, 'duration_sec': -1, 'row_span': -1}, inplace=True)
        else:
            features['is_abnormal'] = False
            features['abnormal_reason'] = ''
        
        features = features[features['valid'].fillna(False)].reset_index(drop=True)
        # 添加 loc 标签
        features['loc'] = loc
        
        all_features_dfs.append(features)

    # 合并所有 loc 的特征
    final_features = pd.concat(all_features_dfs, ignore_index=True)
    final_features = final_features.reset_index(drop=True)
    
    return final_features


def runProcess(df, window_size=10):
    final_features = process_location_data(df)
    loc_groups = final_features['loc'].unique()
    feature_cols = ['max_torque', 'angle_at_max', 'auc', 'slope_rise_late', 'total_angle', 'std_torque', 'row_span']
    bad_results = []

    for loc in loc_groups:
        data = final_features[final_features['loc'] == loc].copy()
        # 滑动窗口标准化
        for col in feature_cols:
            data[f'{col}_z'] = (data[col] - data[col].rolling(window=window_size, min_periods=1).mean()).abs() / (data[col].rolling(window=window_size, min_periods=1).std() + 1e-6)

        clf = IsolationForest(n_estimators=100, contamination=0.005, random_state=1)
        X = data[[f'{f}_z' for f in feature_cols]].fillna(0)
        # X = data[['max_torque', 'angle_at_max', 'auc', 'slope_rise_late', 'total_angle', 'std_torque', 'row_span']].fillna(0)
        # 计算加权异常分数
        y_pred = clf.fit_predict(X)
        anomaly_scores = -clf.score_samples(X)  # 转换为"越异常分数越高"
        data['anomaly_score'] = anomaly_scores
        data['is_anomaly'] = (y_pred == -1).astype(int)
        bad_results.append(data[(data['recommended_action'] != 'Continue') | (data['is_anomaly'] == 1)])

    bad_results = pd.concat(bad_results, ignore_index=True)
    bad_results[['hsgsn', 'time', 'recommended_action', 'is_anomaly', 'loc', 'anomaly_score']]
    return bad_results