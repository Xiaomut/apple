import pandas as pd
import pickle
from datetime import datetime
from algorithms.tcm2 import ScrewdriverReplacementAnalyzer

# 假设你之前已经训练好并保存了 analyzer 对象（可选）
# 如果没有保存，也可以每次都重新初始化（适用于冷启动）

# === Step 1: 初始化或加载已训练的分析器 ===
# 方式一：新建分析器（首次使用）
analyzer = ScrewdriverReplacementAnalyzer(
    window_size=20,
    failure_threshold=3,
    screw_positions=[1, 2, 4, 41],
    top_causes=3,
    contamination=0.02
)

# （可选）方式二：如果你之前保存过训练好的 analyzer
with open('screwdriver_analyzer.pkl', 'rb') as f:
    analyzer = pickle.load(f)
    analyzer.reset()  # 可选择重置历史记录，保留模型


# 调用分析方法（自动复用已有逻辑，无需 re-fit）
result_df = analyzer.analyze_all_positions(new_df)

if result_df.empty:
    print("No result generated. Check input data format.")
else:
    print(f"\n✅ Analysis completed. Total records: {len(result_df)}")
    print(f"🚨 Replacement signals triggered: {result_df['replacement_signal'].sum()}")
    
    # 显示触发更换的记录
    alert_df = result_df[result_df['replacement_signal'] == 1]
    if not alert_df.empty:
        print("\n🚨 Replacement Alerts:")
        print(alert_df[['DateTime', 'screw position', 'SN', 'failure_count', 'anomaly_score']].head())
    else:
        print("\n🟢 No replacement needed.")