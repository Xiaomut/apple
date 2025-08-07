#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   screw_analysis.py
@Time    :   2025/08/06 14:39:08
@Author  :   wangshuai 
@Contact :   wangshuai110@longi.com
@License :   (C)Copyright 2025-2026, WangShuai
'''


import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')
from algorithms.watch import BitWearMonitor


def analyze_screw_data(test_df, train_df=None):
    """
    分析螺丝工序数据
    参数:
        test_df: 测试数据DataFrame
        train_df: 训练数据DataFrame (可选)
    返回: (分析结果字典, 结果文本)
    """
    try:
        # 初始化BitWearMonitor
        monitor = BitWearMonitor()
        
        # 如果有训练数据，先训练模型
        if train_df is not None:
            monitor.train(train_df)
        elif not monitor.is_trained:
            return None, "错误: 未提供训练数据，模型未训练"
        
        # 分析测试数据中的每个螺丝
        results = []
        for name, group in test_df.groupby(['prefix', 'pos']):
            result = monitor.predict_single(group)
            if 'error' not in result:
                results.append(result)
        
        # 生成结果文本
        analysis_text = ""
        warning_count = 0
        for res in results:
            status = res['status']
            if status == 'WARNING':
                warning_count += 1
            analysis_text += f"位置 {res['loc']}: 状态 {status}, 异常分数 {res['anomaly_score']:.2f}\n"
        
        analysis_text += f"\n总计: {len(results)} 个螺丝分析完成, {warning_count} 个警告"
        
        # 生成可视化结果
        image_path = generate_analysis_plot(results)
        
        return {'image_path': image_path, 'results': results}, analysis_text
    
    except Exception as e:
        return None, f"螺丝分析错误: {str(e)}"

def generate_analysis_plot(results):
    """生成螺丝状态分析图"""
    if not results:
        return None
    
    # 准备数据
    locs = [r['loc'] for r in results]
    scores = [r['anomaly_score'] for r in results]
    statuses = [1 if r['status'] == 'WARNING' else 0 for r in results]
    threshold = results[0]['threshold'] if results else 1.5
    
    # 创建图表
    plt.figure(figsize=(10, 5))
    
    # 绘制异常分数
    bars = plt.bar(range(len(locs)), scores, color=np.where(np.array(statuses), 'red', 'green'))
    
    # 添加阈值线
    plt.axhline(y=threshold, color='orange', linestyle='--', label='警告阈值')
    
    # 添加标签和标题
    plt.xticks(range(len(locs)), locs, rotation=45)
    plt.xlabel('螺丝位置')
    plt.ylabel('异常分数')
    plt.title('螺丝状态分析结果')
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = f'temp_screw_analysis_{timestamp}.png'
    plt.savefig(image_path)
    plt.close()
    
    return image_path