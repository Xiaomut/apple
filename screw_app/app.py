from flask import Flask, request, render_template, redirect, url_for, flash, session, jsonify, send_from_directory
import os
import pandas as pd
from datetime import datetime

import sys
sys.path.append(".")
sys.path.append("..")
from algorithms.tcm2 import ScrewdriverReplacementAnalyzer
from algorithms.screw_anlysis import plot_screw_analysis_trend


app = Flask(__name__)
# 配置参数
app.config['UPLOAD_FOLDER'] = 'uploads'  # 上传文件保存路径
app.config['RESULT_FOLDER'] = 'results'  # 分析结果图片保存路径
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 限制上传文件大小10MB
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}  # 允许的文件类型


def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_analyzer():
    """获取或创建全局分析器实例（保持原有逻辑）"""
    analyzer_instance = ScrewdriverReplacementAnalyzer(
        window_size=5,
        failure_threshold=5,
        screw_positions=[4, 1, 2, 41],  # 根据实际数据调整
        top_causes=3,
        contamination=0.01
    )
    return analyzer_instance


def analyze_screw_data(test_df):
    """
    分析螺丝数据（保持原有核心逻辑，优化返回结构）
    参数:
        test_df: 新的测试数据 (DataFrame)
    返回:
        dict: 包含分析结果的字典（success/data/error）
    """
    analyzer = get_analyzer()
    analyzer.reset()  # 重置状态避免历史污染

    try:
        # 执行分析
        result_df = analyzer.analyze_all_positions(test_df)
        if result_df is None or result_df.empty:
            raise ValueError("分析后无有效数据")

        # 获取更换原因
        cause_df = analyzer.get_top_causes_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cause_path = os.path.join(app.config['RESULT_FOLDER'], f"cause_{timestamp}.csv")
        if not cause_df.empty:
            cause_df.sort_values(['time']).to_csv(cause_path, index=False)    #

        # 生成趋势图（保存到结果目录）
        image_path = os.path.join(app.config['RESULT_FOLDER'], f"screw_trend_{timestamp}.png")
        os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)  # 确保目录存在
        plot_screw_analysis_trend(result_df, image_path)  # 假设该函数已实现

        # 构造结果字典
        result = {
            'success': True,
            'data': {
                'image_url': f"/results/screw_trend_{timestamp}.png",  # 前端访问图片的URL
                'cause_url': f"/results/cause_{timestamp}.csv",  # 前端访问图片的URL
                'total_alerts': int(result_df['replacement_signal'].sum()),
                'latest_anomaly_score': round(result_df['weighted_anomaly'].iloc[-1], 4) if len(result_df) > 0 else 0.0,
                'analysis_text': ""
            }
        }

        # 生成文本摘要
        if not cause_df.empty:
            top_cause = cause_df['main_cause'].mode()[0] if not cause_df['main_cause'].empty else "Unknown"
            result['data']['analysis_text'] = (
                f"检测到 {len(cause_df)} 次工具更换信号。主要原因为 '{top_cause}' 超差。建议检查螺丝刀磨损情况。"
            )
            result['data']['recommendation'] = f"建议更换螺丝刀，主因：{top_cause}"
        else:
            result['data']['analysis_text'] = "未检测到异常趋势，螺丝刀状态正常。"

        return result

    except Exception as e:
        print(f"分析过程中发生错误: {str(e)}")
        return {
            'success': False,
            'error': f"分析失败: {str(e)}"
        }


@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传接口"""
    # 检查是否有文件上传
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': '未上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': '空文件名'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': f'不支持的文件类型，仅允许: {ALLOWED_EXTENSIONS}'}), 400

    try:
        # 保存上传文件（可选，根据分析器是否需要原始文件）
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(upload_path)

        # 读取文件为DataFrame
        if file.filename.endswith('.csv'):
            test_df = pd.read_csv(upload_path)
        elif file.filename.endswith('.xlsx'):
            test_df = pd.read_excel(upload_path)
        else:
            return jsonify({'success': False, 'error': '不支持的文件格式'}), 400

        # 执行数据分析
        analysis_result = analyze_screw_data(test_df)

        # 清理临时上传文件（可选）
        # os.remove(upload_path)

        return jsonify(analysis_result)

    except Exception as e:
        print(f"文件处理失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"文件处理失败: {str(e)}"
        }), 500


@app.route('/results/<filename>')
def serve_result(filename):
    """提供分析结果文件（图片或CSV）的访问接口"""
    file_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    # 根据文件扩展名返回不同类型的内容
    if filename.endswith('.png'):
        return send_from_directory(app.config['RESULT_FOLDER'], filename)
    elif filename.endswith('.csv'):
        return send_from_directory(app.config['RESULT_FOLDER'], filename, as_attachment=True)  # 可选：直接下载
    else:
        return jsonify({'error': 'Unsupported file type'}), 400


if __name__ == '__main__':
    # 初始化必要目录
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
    
    # 启动Flask应用（调试模式仅开发时使用）
    app.run(host='0.0.0.0', port=5000, debug=True)