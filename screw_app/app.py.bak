from flask import Flask, request, render_template, redirect, url_for, flash, session
import os
import pandas as pd
import pickle
from werkzeug.utils import secure_filename
from datetime import datetime
import shutil

import sys
sys.path.append(".")
sys.path.append("..")
from algorithms.registration import regist
from algorithms.tcm2 import ScrewdriverReplacementAnalyzer
from algorithms.screw_anlysis import plot_screw_analysis_trend


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # 用于flash消息

# 配置上传文件夹和允许的扩展名
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
ALLOWED_FILE_EXTENSIONS = {'txt', 'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制上传大小为16MB

# 确保上传和结果文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


analyzer_instance = None

def get_analyzer():
    """获取或创建全局分析器实例"""
    global analyzer_instance
    if analyzer_instance is None:
        analyzer_instance = ScrewdriverReplacementAnalyzer(
            window_size=20,
            failure_threshold=3,
            screw_positions=[1, 2, 4, 41],  # 根据实际数据调整
            top_causes=3,
            contamination=0.02
        )
        # 如果有预训练模型，可在此加载
        try:
            with open('screwdriver_analyzer_trained.pkl', 'rb') as f:
                analyzer_instance = pickle.load(f)
        except:
            pass
    return analyzer_instance


def allowed_file(filename, file_type='image'):
    """检查文件扩展名是否合法"""
    allowed_ext = ALLOWED_IMAGE_EXTENSIONS if file_type == 'image' else ALLOWED_FILE_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_ext


def clear_folder(folder_path):
    """清空文件夹内容"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


@app.route('/', methods=['GET', 'POST'])
def index():
    image_paths = session.get('image_paths', [])
    # 检查实际存在的图片文件
    existing_image_paths = []
    for path in image_paths:
        if os.path.exists(path):
            existing_image_paths.append(path)
        else:
            print(f"Warning: Image file not found: {path}")
    
    session['image_paths'] = existing_image_paths

    if request.method == 'POST':
        # 检查是否有文件上传
        if 'train_file' in request.files or 'test_file' in request.files:
            return handle_file_upload(request)
        elif 'image_files' in request.files:
            return handle_image_upload(request)
        elif 'start_calculation' in request.form:
            return handle_calculation(request)
    
    # GET请求或无效POST请求时返回空页面
    return render_template('index.html', image_paths=existing_image_paths)


# 在app.py中添加或修改以下函数
def handle_file_upload(request):
    """处理训练文件和测试文件上传"""
    train_file = request.files.get('train_file')
    test_file = request.files.get('test_file')
    
    # 从session获取已上传的图片路径
    image_paths = session.get('image_paths', [])
    
    # 检查是否至少上传了一个文件
    if (not train_file or train_file.filename == '') and (not test_file or test_file.filename == ''):
        flash('请至少上传一个文件')
        return redirect(url_for('index'))
    
    # 只清除之前的训练/测试文件，保留图片
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith('train_') or filename.startswith('test_'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    uploaded_files = []
    
    # 处理训练文件
    if train_file and train_file.filename != '':
        if allowed_file(train_file.filename, file_type='file'):
            filename = secure_filename(train_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'train_{filename}')
            train_file.save(filepath)
            uploaded_files.append(f'训练文件 {filename}')
        else:
            flash('训练文件类型不支持')
            return redirect(url_for('index'))
    
    # 处理测试文件
    if test_file and test_file.filename != '':
        if allowed_file(test_file.filename, file_type='file'):
            filename = secure_filename(test_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'test_{filename}')
            test_file.save(filepath)
            uploaded_files.append(f'测试文件 {filename}')
        else:
            flash('测试文件类型不支持')
            return redirect(url_for('index'))
    
    flash('上传成功: ' + ', '.join(uploaded_files))
    
    # 保持图片路径不变
    session['image_paths'] = image_paths
    return redirect(url_for('index'))


# 在handle_image_upload函数中修改如下：
def handle_image_upload(request):
    """处理图像上传"""
    files = request.files.getlist('image_files')
    
    if len(files) != 2 or any(f.filename == '' for f in files):
        flash('请选择两张图像')
        return redirect(url_for('index'))
    
    if not all(allowed_file(f.filename) for f in files):
        flash('只支持图片格式: png, jpg, jpeg, gif, bmp')
        return redirect(url_for('index'))
    
    # 只清除之前的图像文件，保留其他文件
    for filename in os.listdir(app.config['UPLOAD_FOLDER']):
        if filename.startswith('image_'):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    image_paths = []
    for i, file in enumerate(files):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'image_{i}_{filename}')
        file.save(filepath)
        image_paths.append(filepath)
        print(filepath)
    
    # 使用session保存图像路径
    session['image_paths'] = image_paths
    flash('两张图像上传成功')
    return redirect(url_for('index'))


def analyze_screw_data(test_df, train_df=None):
    """
    分析螺丝数据：使用 ScrewdriverReplacementAnalyzer 进行预测
    参数:
        test_df: 新的测试数据 (DataFrame)
        train_df: 训练数据（可选，本场景暂不使用）
    返回:
        result_dict: 包含图像路径和文本结果
        analysis_text: 纯文本摘要
    """
    analyzer = get_analyzer()

    # 重置本次分析的记录（避免历史数据污染）
    analyzer.reset()

    try:
        # 执行分析（预测）
        result_df = analyzer.analyze_all_positions(test_df)
        
        if result_df is None or result_df.empty:
            raise ValueError("No valid data after analysis.")

        # 获取更换原因
        cause_df = analyzer.get_top_causes_summary()

        # 生成趋势图
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_filename = f"screw_trend_{timestamp}.png"
        image_path = os.path.join(app.config['RESULT_FOLDER'], image_filename)
        plot_screw_analysis_trend(result_df, image_path)

        # 构造返回结果
        result_dict = {
            'image_path': image_path,
            'total_alerts': int(result_df['replacement_signal'].sum()),
            'latest_anomaly_score': round(result_df['weighted_anomaly'].iloc[-1], 4) if len(result_df) > 0 else 0.0,
        }

        # 文本分析摘要
        if len(cause_df) > 0:
            top_cause = cause_df['main_cause'].mode()[0] if not cause_df['main_cause'].empty else "Unknown"
            analysis_text = (
                f"检测到 {len(cause_df)} 次工具更换信号。"
                f"主要原因为 '{top_cause}' 超差。建议检查螺丝刀磨损情况。"
            )
            result_dict['recommendation'] = f"建议更换螺丝刀，主因：{top_cause}"
        else:
            analysis_text = "未检测到异常趋势，螺丝刀状态正常。"

        result_dict['analysis_text'] = analysis_text

        return result_dict

    except Exception as e:
        print(f"Error in analyze_screw_data: {e}")
        # 返回默认结果
        default_result = {
            'image_path': None,
            'analysis_text': f"分析失败: {str(e)}"
        }
        return default_result, f"分析失败: {str(e)}"


def handle_calculation(request):
    """处理计算请求 - 同时进行图像配准和螺丝状态分析"""
    upload_files = os.listdir(app.config['UPLOAD_FOLDER'])
    if not upload_files:
        flash('请先上传文件或图像')
        return redirect(url_for('index'))
    
    clear_folder(app.config['RESULT_FOLDER'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 初始化结果变量
    registered_image_path = None
    screw_analysis_result = None
    
    # 检查并处理图像配准
    image_paths = [os.path.join(app.config['UPLOAD_FOLDER'], f) for f in upload_files if f.startswith('image_')]
    
    if len(image_paths) == 2:
        try:
            registered_image_path = regist(*image_paths)
            if registered_image_path:
                result_filename = f'registered_{timestamp}.png'
                result_filepath = os.path.join(app.config['RESULT_FOLDER'], result_filename)
                shutil.move(registered_image_path, result_filepath)
                registered_image_path = result_filepath
                flash('图像配准成功')
        except Exception as e:
            flash(f'图像配准失败: {str(e)}')
    
    # 检查并处理螺丝数据文件
    train_files = [f for f in upload_files if f.startswith('train_')]
    test_files = [f for f in upload_files if f.startswith('test_')]
    print(test_files)
    
    if test_files:
        try:
            test_file_path = os.path.join(app.config['UPLOAD_FOLDER'], test_files[0])
            test_df = pd.read_csv(test_file_path)  # 确保此函数能正确读取 CSV
            
            # 注意：当前逻辑中，train_df 暂不用于训练（无监督分析）
            # 如果未来要做监督学习，可使用 train_df
            screw_analysis_result = analyze_screw_data(test_df, train_df=None)
            print(screw_analysis_result)
            flash('螺丝刀状态分析成功')
        
        except Exception as e:
            flash(f'螺丝数据分析失败: {str(e)}')
            import traceback
            print(traceback.format_exc())
    
    return render_template('index.html',
                         image_paths=image_paths,
                         registered_image_path=registered_image_path,
                         screw_analysis_result=screw_analysis_result)


if __name__ == '__main__':
    app.run(debug=True)
    # df = pd.read_csv(r'screw_app\static\uploads\screw_copy.csv')
    # screw_analysis_result, screw_analysis_text = analyze_screw_data(df)