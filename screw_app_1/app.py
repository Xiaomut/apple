#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2025/8/14 18:34
# @Author  : Maoheng
# @Email   : maoheng@longi.com
# @File    : app.py
# @Software: PyCharm
from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# 配置：上传文件夹、允许的文件扩展名
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 限制上传文件大小为 10MB


def allowed_file(filename):
    """检查文件扩展名是否合法"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def beautify_sales_data(df):
    df.columns = ['工序', '单位', '合计', '第一周', '第二周', '第三周', '第四周', '第五周', '第六周']
    # 美化数值显示
    df['合计'] = df['合计'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第一周'] = df['第一周'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第二周'] = df['第二周'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第三周'] = df['第三周'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第四周'] = df['第四周'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第五周'] = df['第五周'].apply(lambda x: f"{x / 10000:,.0f}")
    df['第六周'] = df['第六周'].apply(lambda x: f"{x / 10000:,.0f}")

    return df


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    table_html = None
    filename = None
    error = None
    print(request.method)
    if request.method == 'POST':
        # 检查是否有文件上传
        if 'file' not in request.files:
            error = "未选择文件"
            return render_template('index.html', table_html=table_html, filename=filename, error=error)

        file = request.files['file']

        # 检查文件名是否为空
        if file.filename == '':
            error = "未选择文件"
            return render_template('index.html', table_html=table_html, filename=filename, error=error)

        # 检查文件类型
        if file and allowed_file(file.filename):
            # 安全处理文件名
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # 创建上传目录（如果不存在）
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

            try:
                # 保存文件
                file.save(filepath)

                # 读取Excel数据
                if filename.endswith('.csv'):
                    df = pd.read_csv(filepath)
                elif filename.endswith('.xls') or filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath, engine='openpyxl')

                # 美化数据
                # cleaned_df = beautify_sales_data(df)

                # 转换为HTML表格（带Bootstrap样式）
                table_html = df.to_html(
                    classes='table table-striped table-bordered table-hover',
                    index=True,
                    border=0,
                    escape=False,
                    header="true"
                )

            except Exception as e:
                error = f"处理文件时出错: {str(e)}"
                # 删除无效文件
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            error = "不支持的文件类型，请上传Excel文件(.csv、 .xlsx或.xls)"
        image_url = f'/static/uploads/IMG20250815-162348.png'
        return render_template('index.html',  image_url=image_url, filename=filename, error=error)
    else:
        # GET 请求返回上传页面
        return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)  # 开发模式开启调试（生产环境需关闭）
