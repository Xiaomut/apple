#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   watch.py
@Time    :   2025/08/06 13:16:34
@Author  :   wangshuai 
@Contact :   wangshuai110@longi.com
@License :   (C)Copyright 2025-2026, WangShuai
'''


# model.py
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class BitWearMonitor:
    def __init__(self):
        self.models = {}  # 每个 loc 保存其最近历史数据和统计量
        self.is_trained = False
        self.last_trained = None
        self.feature_cols = [
            'max_torque', 'auc', 'slope_rise_late', 'total_angle', 'std_torque', 'duration_sec', 'row_span'
        ]
        self.weights = {
            'max_torque': 0.8,
            'auc': 1.2,
            'slope_rise_late': 1.3,
            'total_angle': 1.0,
            'std_torque': 0.9,
            'duration_sec': 1.1,
            'row_span': 1.0
        }
        self.window_size = 10
        self.threshold = 1.5

    def extract_features(self, df_group):
        """提取单颗螺丝特征"""
        valid_mask = (df_group['angle'] > 0) & (df_group['torque'] > 0)
        if not valid_mask.any():
            return None

        data = df_group[valid_mask].copy().sort_values('angle')

        angle = data['angle'].values
        torque = data['torque'].values
        row_idx = data['row_in_file'].values
        times = pd.to_datetime(data['datetime']).values

        if len(angle) < 2:
            return None

        max_torque = torque.max()
        angle_at_max = angle[np.argmax(torque)]
        auc = np.trapz(torque, angle)

        peak_idx = np.argmax(torque)
        if peak_idx < 2:
            slope_rise_late = 0.0
        else:
            mid_angle = (angle[0] + angle[peak_idx]) / 2
            late_mask = (angle >= mid_angle) & (angle <= angle[peak_idx])
            late_idx = np.where(late_mask)[0]
            if len(late_idx) >= 2:
                i1, i2 = late_idx[0], late_idx[-1]
                slope_rise_late = (torque[i2] - torque[i1]) / (angle[i2] - angle[i1] + 1e-6)
            else:
                slope_rise_late = 0.0

        total_angle = angle[-1] - angle[0]
        std_torque = np.std(torque)
        duration_sec = (times[-1] - times[0]) / np.timedelta64(1, 's')
        row_span = row_idx[-1] - row_idx[0]

        return {
            'max_torque': max_torque,
            'angle_at_max': angle_at_max,
            'auc': auc,
            'slope_rise_late': slope_rise_late,
            'total_angle': total_angle,
            'std_torque': std_torque,
            'duration_sec': duration_sec,
            'row_span': row_span,
            'loc': int(df_group['loc_clean'].iloc[0]),
            'timestamp': times[0]
        }

    def train(self, df):
        """训练：从历史数据中提取每 loc 的特征并缓存"""
        df = df.copy()
        df['loc_clean'] = df['loc'].astype(str).str.replace('Loc', '').astype(int)
        df['screw_id'] = df['prefix'] + '_' + df['pos']

        # # 合并 loc4 和 loc41  Not Use!
        # mask_4_41 = df['loc_clean'].isin([4, 41])
        # df.loc[mask_4_41, 'screw_id'] = df.loc[mask_4_41, 'prefix'] + '_pos4_loc4'

        all_features = []
        for name, group in df.groupby('screw_id'):
            feat = self.extract_features(group)
            if feat:
                feat['screw_id'] = name
                all_features.append(feat)

        if not all_features:
            raise ValueError("No valid screws found in training data.")

        feature_df = pd.DataFrame(all_features)
        feature_df['timestamp'] = pd.to_datetime(feature_df['timestamp'])

        # 按 loc 分组缓存
        for loc in feature_df['loc'].unique():
            self.models[loc] = feature_df[feature_df['loc'] == loc].sort_values('timestamp').reset_index(drop=True)

        self.is_trained = True
        self.last_trained = datetime.now()
        print(f"✅ Model trained on {len(feature_df)} screws across {len(self.models)} loc groups.")

    def predict_single(self, new_data):
        """预测单颗螺丝"""
        if not self.is_trained:
            return {"error": "Model not trained yet."}

        feat = self.extract_features(new_data)
        if not feat:
            return {"error": "Invalid or empty screw data."}

        loc = feat['loc']
        if loc not in self.models:
            return {"error": f"No training data for loc={loc}"}

        history = self.models[loc].tail(self.window_size)
        if len(history) < 3:
            return {"error": f"Not enough history for loc={loc}"}

        z_scores = []
        details = {}

        for col in self.feature_cols:
            mean = history[col].mean()
            std = history[col].std()
            value = feat[col]
            z = 0 if std == 0 else abs(value - mean) / std
            weighted_z = z * self.weights[col]
            z_scores.append(weighted_z)
            details[col] = {
                "value": round(value, 4),
                "mean": round(mean, 4),
                "std": round(std, 4),
                "z_score": round(z, 3),
                "weighted_z": round(weighted_z, 3)
            }

        anomaly_score = sum(z_scores) / sum(self.weights.values())
        is_warning = anomaly_score > self.threshold

        return {
            "loc": loc,
            "anomaly_score": round(anomaly_score, 3),
            "threshold": self.threshold,
            "status": "OK" if not is_warning else "WARNING",
            "suggest_replace": is_warning,
            "features": details
        }