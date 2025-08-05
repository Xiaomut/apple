import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure, feature
from skimage.feature import graycomatrix, graycoprops

import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def register_and_compare(before_path, after_path):
    # ---------------------- 步骤1：读取图像并灰度化 ----------------------
    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    if before is None or after is None:
        raise ValueError("无法读取图像，请检查路径是否正确！")
    
    before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

    # ---------------------- 步骤2：特征检测与描述（SIFT） ----------------------
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        # 如果没有SIFT，使用ORB替代
        sift = cv2.ORB_create()
    kp_before, des_before = sift.detectAndCompute(before_gray, None)  # 使用前的特征点和描述符
    kp_after, des_after = sift.detectAndCompute(after_gray, None)    # 使用后的特征点和描述符

    # ---------------------- 步骤3：特征匹配（FLANN） ----------------------
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # 增加checks提高匹配精度
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des_before, des_after, k=2)  # 每个特征点找2个最近邻

    # ---------------------- 步骤4：筛选优质匹配点（Lowe's Ratio Test） ----------------------
    good_matches = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:  # 保留距离比小于0.7的匹配点
            good_matches.append(m)

    if len(good_matches) < 4:  # 至少需要4个点估计变换矩阵
        raise RuntimeError("优质匹配点不足，无法配准！")

    # ---------------------- 步骤5：使用RANSAC估计变换矩阵 ----------------------
    src_pts = np.float32([kp_before[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_after[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 估计单应性矩阵（适用于旋转、平移、缩放场景）
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)  # 5.0是RANSAC的重投影误差阈值
    mask = mask.ravel().tolist()  # 转换为列表以便筛选

    # ---------------------- 步骤6：应用变换对齐图像 ----------------------
    h, w = before_gray.shape
    aligned_after = cv2.warpPerspective(after, M, (w, h))  # 将使用后的图像对齐到使用前的坐标系

    # ---------------------- 步骤7：计算差异图（突出磨损区域） ----------------------
    # 转换为RGB以便显示差异（原图是BGR格式）
    before_rgb = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    aligned_after_rgb = cv2.cvtColor(aligned_after, cv2.COLOR_BGR2RGB)
    
    # 计算绝对差异（像素级差异）
    diff = cv2.absdiff(before_rgb, aligned_after_rgb)
    # 增强差异显示（可选：用阈值过滤微小差异）
    _, diff_thresh = cv2.threshold(cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY), 30, 255, cv2.THRESH_BINARY)
    diff_highlight = cv2.bitwise_and(diff, diff, mask=diff_thresh)

    # ---------------------- 步骤8：可视化结果 ----------------------
    plt.figure(figsize=(20, 10))

    # 子图1：使用前图像
    plt.subplot(2, 3, 1)
    plt.imshow(before_rgb)
    plt.title('使用前批杆')
    plt.axis('off')

    # 子图2：使用后原始图像
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(after, cv2.COLOR_BGR2RGB))
    plt.title('使用后批杆（原始）')
    plt.axis('off')

    # 子图3：使用后对齐图像
    plt.subplot(2, 3, 3)
    plt.imshow(aligned_after_rgb)
    plt.title('使用后批杆（对齐后）')
    plt.axis('off')

    # 子图4：差异图（原始）
    plt.subplot(2, 3, 4)
    plt.imshow(diff)
    plt.title('原始差异图')
    plt.axis('off')

    # 子图5：增强差异图（仅显示显著磨损）
    plt.subplot(2, 3, 5)
    plt.imshow(diff_highlight)
    plt.title('增强差异图（磨损区域）')
    plt.axis('off')

    # 子图6：差异叠加（半透明显示）
    overlay = aligned_after_rgb.copy()
    overlay[diff_thresh == 255] = [255, 0, 0]  # 磨损区域标记为红色
    plt.subplot(2, 3, 6)
    plt.imshow(overlay)
    plt.title('磨损区域叠加显示')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return aligned_after, diff_highlight


# ---------------------- 执行配准与对比 ----------------------
if __name__ == "__main__":
    # 替换为实际图像路径（假设before.jpg是使用前，after.jpg是使用后）
    aligned_img, diff_img = register_and_compare(r"datas\imgs\items\new.jpg", r"datas\imgs\items\old.jpg")