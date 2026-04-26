"""
高速公路车辆与车道检测 —— IPM(逆透视) + HSV 多通道
─────────────────────────────────────────────────────────────
整体流程：
  原图 → HSV 通道分解 → 白/黄车道线提取
       ↓
  Canny + HoughLinesP → RANSAC 估计消失点（失败时回退默认梯形）
       ↓
  按消失点构造 IPM 源梯形 → cv2.warpPerspective → 鸟瞰图
       ↓
  鸟瞰图列直方图寻峰 → 车道边界 x 坐标
       ↓
  HSV 多通道(暗色 ∪ 高饱和) + 凸包填充 → 车辆 bbox
       ↓
  车辆"底中点"经 H 反投影到鸟瞰图 → 落入哪条车道
       ↓
  黄色实线画所有车道；红色粗线画最忙车道；输出+调试拼图
"""

import os
import cv2
import numpy as np
from collections import defaultdict

# ═════════════════════════════════════════════════════════════
# 配置
# ═════════════════════════════════════════════════════════════
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
IMG_PATH    = os.path.join(PROJECT_ROOT, "image", "cim019.jpg")
OUT_PATH    = os.path.join(PROJECT_ROOT, "result.jpg")
DEBUG_DIR   = os.path.join(PROJECT_ROOT, "debug")
os.makedirs(DEBUG_DIR, exist_ok=True)

BEV_W, BEV_H = 600, 800   # 鸟瞰图尺寸


# ═════════════════════════════════════════════════════════════
# 工具函数
# ═════════════════════════════════════════════════════════════
def line_intersection(l1, l2):
    """两条线段所在直线的交点；平行/重合返回 None"""
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-6:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def find_vanishing_point(lines, img_shape, iters=300, dist_thr=10):
    """
    RANSAC 估计消失点：随机抽两条线段，求其延长线交点 vp，
    再统计有多少条线段的延长线"几乎"通过 vp（点-线距离 < dist_thr），
    支持最多的 vp 即返回值。
    """
    h, w = img_shape
    if lines is None or len(lines) < 4:
        return None

    cand = []
    for ln in lines:
        x1, y1, x2, y2 = ln[0]
        if np.hypot(x2 - x1, y2 - y1) < 30:
            continue
        # 排除接近水平的线（车道线在透视图中绝不会水平）
        if abs(y2 - y1) < 0.3 * abs(x2 - x1):
            continue
        cand.append((float(x1), float(y1), float(x2), float(y2)))

    if len(cand) < 4:
        return None

    rng = np.random.default_rng(42)
    best_vp, best_inl = None, 0
    for _ in range(iters):
        i, j = rng.choice(len(cand), 2, replace=False)
        vp = line_intersection(cand[i], cand[j])
        if vp is None:
            continue
        vx, vy = vp
        # 消失点合理范围（可在图外但不能离谱）
        if not (-w < vx < 2 * w and -2 * h < vy < h):
            continue

        inl = 0
        for x1, y1, x2, y2 in cand:
            num = abs((y2 - y1) * vx - (x2 - x1) * vy + x2 * y1 - x1 * y2)
            den = np.hypot(y2 - y1, x2 - x1)
            if den > 1e-6 and num / den < dist_thr:
                inl += 1
        if inl > best_inl:
            best_inl, best_vp = inl, (vx, vy)

    if best_inl < 4 or best_vp is None:
        return None
    # 进一步校验：高速摄像头消失点应在图像上半区
    if best_vp[1] > 0.55 * h:
        return None
    return best_vp


def build_ipm_src(vp, h, w):
    """
    根据消失点构造 IPM 源梯形；vp 为 None 时回退默认梯形。
    顶点顺序：[左下, 左上, 右上, 右下]
    """
    if vp is None:
        return np.array([
            [int(w * 0.02), h],
            [int(w * 0.30), int(h * 0.10)],
            [int(w * 0.78), int(h * 0.10)],
            [int(w * 0.98), h],
        ], dtype=np.float32)

    vx, vy = vp
    # y_top 取在消失点稍下方（保证有可观察的路面）
    y_top = max(int(vy + (h - vy) * 0.20), int(h * 0.05))
    y_top = min(y_top, int(h * 0.45))

    # 沿 (vp → 图像底部左/右) 直线，求 y=y_top 时的 x 坐标
    t = (y_top - vy) / max(h - vy, 1e-6)
    bot_lx, bot_rx = int(w * 0.02), int(w * 0.98)
    top_lx = vx + t * (bot_lx - vx)
    top_rx = vx + t * (bot_rx - vx)

    return np.array([
        [bot_lx, h],
        [int(top_lx), y_top],
        [int(top_rx), y_top],
        [bot_rx, h],
    ], dtype=np.float32)


def find_peaks_1d(values, min_h, min_dist, min_prom=0):
    """
    1D 滑窗寻峰：
      - 局部最大（min_dist 邻域内最高）
      - 绝对高度 ≥ min_h
      - 相邻峰间距 ≥ min_dist
      - prominence ≥ min_prom（与左/右最近低谷的高度差，过滤虚弱凸起）
    """
    peaks = []
    n = len(values)
    for x in range(n):
        if values[x] < min_h:
            continue
        lo, hi = max(0, x - min_dist), min(n, x + min_dist + 1)
        if values[x] < values[lo:hi].max():
            continue
        if min_prom > 0:
            wlo, whi = max(0, x - 80), min(n, x + 80)
            local_min = min(values[wlo:x].min() if x > wlo else values[x],
                            values[x + 1:whi].min() if x + 1 < whi else values[x])
            if values[x] - local_min < min_prom:
                continue
        if not peaks or x - peaks[-1] >= min_dist:
            peaks.append(x)
    return peaks


def show(image, title="Result"):
    """显示图像；OpenCV 无 GUI 时回退到 matplotlib"""
    try:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        import matplotlib.pyplot as plt
        plt.figure(title, figsize=(12, 7))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(title); plt.axis("off"); plt.tight_layout(); plt.show()


# ═════════════════════════════════════════════════════════════
# 0. 读取图像
# ═════════════════════════════════════════════════════════════
img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"无法读取图像：{IMG_PATH}")

output = img.copy()
h, w = img.shape[:2]

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
H_ch, S_ch, V_ch = cv2.split(hsv)


# ═════════════════════════════════════════════════════════════
# 1. 车道线颜色提取（白色虚线 + 黄色边线/中央隔离带）
# ═════════════════════════════════════════════════════════════
white_mask  = ((S_ch < 50)  & (V_ch > 180)).astype(np.uint8) * 255
yellow_mask = ((H_ch >= 15) & (H_ch <= 35) &
               (S_ch > 80)  & (V_ch > 80)).astype(np.uint8) * 255
lane_color_mask = cv2.bitwise_or(white_mask, yellow_mask)


# ═════════════════════════════════════════════════════════════
# 2. Canny + Hough → RANSAC 估计消失点
# ═════════════════════════════════════════════════════════════
edges = cv2.Canny(blur, 50, 150)

# 给 Canny 加个简单的全图掩膜（剔除最顶部 5% 边缘噪声）
roi_top_clip = np.zeros((h, w), dtype=np.uint8)
cv2.rectangle(roi_top_clip, (0, int(h * 0.05)), (w, h), (255,), -1)
edges_roi = cv2.bitwise_and(edges, roi_top_clip)

lines = cv2.HoughLinesP(edges_roi, 1, np.pi / 180,
                        threshold=60, minLineLength=60, maxLineGap=30)

vp = find_vanishing_point(lines, (h, w))
if vp is None:
    print("[VP] 估计失败，回退默认梯形")
else:
    print(f"[VP] 估计成功：x={vp[0]:.1f}, y={vp[1]:.1f}")


# ═════════════════════════════════════════════════════════════
# 3. 构建 IPM
# ═════════════════════════════════════════════════════════════
src_quad = build_ipm_src(vp, h, w)
dst_quad = np.array([
    [0,     BEV_H],
    [0,     0    ],
    [BEV_W, 0    ],
    [BEV_W, BEV_H],
], dtype=np.float32)

H_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
H_inv = np.linalg.inv(H_mat)


# ═════════════════════════════════════════════════════════════
# 4. 鸟瞰图：车道边界（列直方图寻峰）
# ═════════════════════════════════════════════════════════════
img_bev  = cv2.warpPerspective(img,             H_mat, (BEV_W, BEV_H))
lane_bev = cv2.warpPerspective(lane_color_mask, H_mat, (BEV_W, BEV_H))

# 沿竖直方向膨胀，把虚线段连成连续的竖条
v_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 51))
lane_bev_d = cv2.dilate(lane_bev, v_kernel, iterations=1)

# 列直方图 + 平滑
col_hist = np.sum(lane_bev_d > 0, axis=0).astype(np.float32)
ksz = 21
smooth = np.convolve(col_hist, np.ones(ksz) / ksz, mode="same")

peaks = find_peaks_1d(smooth, min_h=BEV_H * 0.06, min_dist=50,
                      min_prom=BEV_H * 0.04) # type: ignore
# 剔除紧贴 BEV 左右边缘的伪峰（一般是 warp 边界伪迹）
peaks = [p for p in peaks if 10 < p < BEV_W - 10]
print(f"[Lane] BEV 内侧白虚线峰值数：{len(peaks)} → {peaks}")


# ═════════════════════════════════════════════════════════════
# 4.5 外侧 2 根实线 —— 方案 B（在原图中检测，避免梯形裁剪）
# ─────────────────────────────────────────────────────────────
# 内侧 4 根白虚线在 BEV 列直方图中很稳；但黄色隔离带与右路肩白实线
# 贴在 IPM 梯形顶部之外，warp 后信号过弱。改在原图坐标系用 fitLine
# 直接检出整条实线，再把下端点投影到 BEV 得到 bev_x，与内侧 4 峰
# 合并成 6 个车道边界（即 5 车道）。
# ═════════════════════════════════════════════════════════════
def fit_solid_line(contour, h_):
    """对轮廓 fitLine，端点延伸到 [0.05h, h]，返回 (p_top, p_bot)。"""
    [vx_, vy_, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0,
                                     0.01, 0.01).flatten()
    if abs(vy_) < 1e-3:
        return None
    t_bot = (h_       - y0) / vy_
    t_top = (0.05 * h_ - y0) / vy_
    p_bot = (int(x0 + vx_ * t_bot), int(y0 + vy_ * t_bot))
    p_top = (int(x0 + vx_ * t_top), int(y0 + vy_ * t_top))
    return (p_top, p_bot)

# 4.5.1 黄色隔离带：取最大黄色轮廓做 fitLine
yellow_pts = None
cnts_yo, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                              cv2.CHAIN_APPROX_SIMPLE)
if cnts_yo:
    c_yo = max(cnts_yo, key=cv2.contourArea)
    if cv2.contourArea(c_yo) > 800:
        yellow_pts = fit_solid_line(c_yo, h)

# 4.5.2 右路肩白实线：在 white_mask 中筛选"右侧 + 竖向长条 + 触及画面下半 +
#       指向 VP + 终点 x 在图像合理范围"的轮廓
shoulder_pts = None
if vp is not None:
    vpx, vpy = vp
    cnts_w, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    best_score, best_c = -1.0, None
    for c in cnts_w:
        x, y, bw, bh = cv2.boundingRect(c)
        if x + bw < 0.55 * w:                  # 整体偏右
            continue
        if y + bh < 0.6 * h:                   # 必须延伸到下半幅，排除墙体上半
            continue
        if bh < 150:                           # 竖直跨度足够
            continue
        if bh < 0.7 * bw:                      # 拒绝偏水平的墙体长条
            continue
        [vx_, vy_, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0,
                                         0.01, 0.01).flatten()
        if abs(vy_) < 1e-3:
            continue
        if abs(vy_ * (vpx - x0) - vx_ * (vpy - y0)) > 30:
            continue
        x_h = x0 + vx_ * ((h - y0) / vy_)
        if not (0.6 * w < x_h < 1.3 * w):      # 终点应贴近图像右边缘
            continue
        # 偏好更长更靠右
        score = float(bh + 0.3 * x_h)
        if score > best_score:
            best_score, best_c = score, c
    if best_c is not None:
        shoulder_pts = fit_solid_line(best_c, h)

# 4.5.3 投影到 BEV，与内侧 4 峰合并成总边界
def bottom_to_bev_x(pts):
    if pts is None:
        return None
    src = np.array([[pts[1]]], dtype=np.float32)  # p_bot
    return float(cv2.perspectiveTransform(src, H_mat)[0][0][0])

bev_x_yellow   = bottom_to_bev_x(yellow_pts)
bev_x_shoulder = bottom_to_bev_x(shoulder_pts)

# (bev_x, p_top, p_bot)：内侧的 p_top/p_bot 由 BEV 反投影得到，
# 外侧两条直接用图像空间端点，避免 perspectiveTransform 在梯形外
# 区域产生不可控坐标。
boundaries = []
def bev_x_to_orig_line(bev_x):
    pts_bev  = np.array([[[bev_x, 0]], [[bev_x, BEV_H]]], dtype=np.float32)
    pts_orig = cv2.perspectiveTransform(pts_bev, H_inv)
    return tuple(pts_orig[0][0].astype(int)), tuple(pts_orig[1][0].astype(int))

for px in peaks:
    p_top, p_bot = bev_x_to_orig_line(px)
    boundaries.append((float(px), p_top, p_bot, "inner"))
if yellow_pts is not None and bev_x_yellow is not None:
    boundaries.append((bev_x_yellow, yellow_pts[0], yellow_pts[1], "outer"))
if shoulder_pts is not None and bev_x_shoulder is not None:
    boundaries.append((bev_x_shoulder, shoulder_pts[0], shoulder_pts[1], "outer"))
boundaries.sort(key=lambda b: b[0])

lane_boundaries_bev  = [b[0] for b in boundaries]
lane_boundaries_orig = [(b[1], b[2]) for b in boundaries]
num_lanes = max(0, len(lane_boundaries_bev) - 1)
print(f"[Lane] 黄线 bev_x={bev_x_yellow}, 右路肩 bev_x={bev_x_shoulder}")
print(f"[Lane] 总边界数：{len(lane_boundaries_bev)}  车道数：{num_lanes}")


# ═════════════════════════════════════════════════════════════
# 5. 车辆检测（HSV 多通道 + 凸包填充）
# ═════════════════════════════════════════════════════════════
# 车辆 ROI：聚焦右侧主道——左边界绕开中央隔离带，右边界绕开高墙
veh_roi = np.zeros((h, w), dtype=np.uint8)
cv2.fillPoly(veh_roi, [np.array([
    [int(w * 0.28), h],
    [int(w * 0.36), int(h * 0.05)],
    [int(w * 0.65), int(h * 0.05)],
    [int(w * 0.78), h],
], dtype=np.int32)], (255,))

# 自动剔除"中央黄色隔离带"——黄色护栏只覆盖下半段，但上半段灰色
# 混凝土也属于隔离带；用 fitLine 沿其主轴延伸到消失点附近，
# 形成一条粗的"中央带"从 ROI 中扣除
median_seed = cv2.dilate(yellow_mask,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                         iterations=2)
median_band = np.zeros_like(veh_roi)
cnts_y, _ = cv2.findContours(median_seed, cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)
for c in cnts_y:
    if cv2.contourArea(c) < 4000:
        continue
    # 先把已检出的黄色块画上
    cv2.drawContours(median_band, [c], -1, (255,), -1)

    # 沿黄色块的主方向延伸到画面顶部，覆盖灰色混凝土段
    [vx_, vy_, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    if abs(vy_) < 1e-3:
        continue
    pts = c.reshape(-1, 2)
    t = (pts[:, 1] - y0) / vy_
    t_min, t_max = float(t.min()), float(t.max())
    span = max(abs(t_max - t_min), 200)
    # 哪个方向是"上"取决于 vy_ 的符号
    if vy_ < 0:
        t_top = t_max + span      # 继续朝消失点（y 减小）方向延伸
        t_bot = t_min
    else:
        t_top = t_min - span
        t_bot = t_max
    p_top = (int(x0 + vx_ * t_top), int(y0 + vy_ * t_top))
    p_bot = (int(x0 + vx_ * t_bot), int(y0 + vy_ * t_bot))
    cv2.line(median_band, p_top, p_bot, (255,), 60, cv2.LINE_AA)

veh_roi = cv2.bitwise_and(veh_roi, cv2.bitwise_not(median_band))

# 5.1 暗色车（V 显著低于路面）：取分位数与绝对上限的较小者，
#     避免路面阴影 / 沥青本身被当作车
v_road = V_ch[veh_roi > 0]
v_th   = int(min(np.percentile(v_road, 14), 115))
v_th   = max(v_th, 60)
dark_mask = ((V_ch < v_th) & (veh_roi > 0)).astype(np.uint8) * 255

# 5.2 彩色车（S 高，且非黄/白车道线）
color_mask = ((S_ch > 60) & (veh_roi > 0)).astype(np.uint8) * 255
color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(yellow_mask))
color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))

# 5.3 边缘密度：车辆有丰富边缘（车身轮廓、车窗、车灯），
#     用 Sobel 幅值膨胀作为"高纹理区域"掩膜，与暗色 mask 取交，
#     可有效剔除"暗但平滑"的路面阴影
sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
edge_strong = (sobel_mag > 35).astype(np.uint8) * 255
edge_density = cv2.dilate(edge_strong,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                          iterations=2)

dark_mask  = cv2.bitwise_and(dark_mask,  edge_density)
color_mask = cv2.bitwise_and(color_mask, edge_density)

# 5.3.5 亮色车分支：白/银车 V 很高、S 很低，前两路都漏。
# 直接 (V 高 ∩ S 低 ∩ ROI) 会把白色车道线一并抓进来，
# 因此用已检出的 6 条 lane_boundaries_orig 构造一条"车道线走廊"
# 先从亮色 mask 中剔除，仅留下真正的车身。
dash_corridor = np.zeros((h, w), dtype=np.uint8)
for (p1, p2) in lane_boundaries_orig:
    cv2.line(dash_corridor, p1, p2, (255,), 20, cv2.LINE_AA)

bright_mask = ((V_ch > 210) & (S_ch < 60) & (veh_roi > 0)).astype(np.uint8) * 255
bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(dash_corridor))
bright_mask = cv2.bitwise_and(bright_mask, edge_density)

vehicle_mask = cv2.bitwise_or(dark_mask, color_mask)
vehicle_mask = cv2.bitwise_or(vehicle_mask, bright_mask)

# 5.4 形态学：保守的闭运算，避免合并相邻车辆
k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_OPEN,  k3, iterations=1)
vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, k5, iterations=2)

# 5.5 选择性凸包填充
contours_raw, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
filled = np.zeros_like(vehicle_mask)
for cnt in contours_raw:
    a = cv2.contourArea(cnt)
    if a < 50:
        continue
    if a > 6000:                                  # 大轮廓 → 不做凸包
        cv2.drawContours(filled, [cnt], -1, (255,), -1)
    else:                                         # 小/中轮廓 → 凸包合并碎片
        cv2.drawContours(filled, [cv2.convexHull(cnt)], -1, (255,), -1)

# 5.6 连通域 + 透视感知过滤
n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

vehicles = []
for i in range(1, n_comp):
    x, y, bw, bh, area = stats[i]
    if bw == 0 or bh == 0:
        continue
    aspect = bw / float(bh)
    rel_y  = (y + bh / 2.0) / h
    # 透视感知的面积阈值——远处车非常小（几十像素），近处车可达数千。
    # 顶部用 40 起步，便于捕获消失点附近的小目标；近处仍用 1500 上调上限。
    min_area = 40  + int(1500 * rel_y)
    max_area = 3000 + int(70000 * rel_y)
    if not (min_area < area < max_area):
        continue
    if not (0.4 < aspect < 3.0):
        continue
    if bw > w * 0.45 or bh > h * 0.55:
        continue
    if area / float(bw * bh) < 0.30:
        continue

    blob = (labels == i)
    h_in, s_in = H_ch[blob], S_ch[blob]
    yellow_ratio = float(((h_in >= 15) & (h_in <= 35) & (s_in > 80)).mean())
    if yellow_ratio > 0.40:           # 黄色护栏排除
        continue

    vehicles.append((int(x), int(y), int(bw), int(bh)))

print(f"[Vehicle] 检出车辆数：{len(vehicles)}")


# ═════════════════════════════════════════════════════════════
# 6. 车辆归属车道（投影到鸟瞰图）
# ═════════════════════════════════════════════════════════════
lane_counts = defaultdict(int)
veh_bev_xs = []
for (x, y, bw, bh) in vehicles:
    bot_center = np.array([[[x + bw / 2.0, y + bh]]], dtype=np.float32)
    bev_pt = cv2.perspectiveTransform(bot_center, H_mat)[0][0]
    bev_x  = float(bev_pt[0])
    veh_bev_xs.append(bev_x)
    if not (0 <= bev_x < BEV_W):
        continue
    for li in range(len(lane_boundaries_bev) - 1):
        if lane_boundaries_bev[li] <= bev_x < lane_boundaries_bev[li + 1]:
            lane_counts[li] += 1
            break


# ═════════════════════════════════════════════════════════════
# 7. 可视化（黄线=所有车道, 红粗线=最忙车道, 绿框=车辆）
# ═════════════════════════════════════════════════════════════
# 黄色实线：所有车道边界（外侧 2 条直接用图像空间端点）
for (p1, p2) in lane_boundaries_orig:
    cv2.line(output, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

# 红色粗线：最忙车道左右边界
busiest_msg = "无"
if lane_counts and num_lanes > 0:
    busiest = max(lane_counts, key=lambda k: lane_counts[k])
    p1, p2 = lane_boundaries_orig[busiest]
    p3, p4 = lane_boundaries_orig[busiest + 1]
    cv2.line(output, p1, p2, (0, 0, 255), 4, cv2.LINE_AA)
    cv2.line(output, p3, p4, (0, 0, 255), 4, cv2.LINE_AA)
    busiest_msg = f"车道 {busiest + 1}（{lane_counts[busiest]} 辆）"
    print(f"[Busiest] {busiest_msg}")

# 绿色框：车辆
for (x, y, bw, bh) in vehicles:
    cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

cv2.putText(output, f"Vehicles: {len(vehicles)}  Lanes: {num_lanes}",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2,
            cv2.LINE_AA)

cv2.imwrite(OUT_PATH, output)
print(f"[Output] 结果已保存：{OUT_PATH}")


# ═════════════════════════════════════════════════════════════
# 8. 调试可视化（单图 + 总览拼图）
# ═════════════════════════════════════════════════════════════
def to_bgr(mask):
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) if mask.ndim == 2 else mask

def label(image_bgr, text):
    out = image_bgr.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 255), 1, cv2.LINE_AA)
    return out

def fit(image_bgr, tw, th):
    """等比缩放，居中填充到 (tw, th)"""
    ih, iw = image_bgr.shape[:2]
    s = min(tw / iw, th / ih)
    nw, nh = int(iw * s), int(ih * s)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = cv2.resize(image_bgr, (nw, nh))
    return canvas

# 8.1 IPM 梯形 + 消失点 标注图
trap_vis = img.copy()
cv2.polylines(trap_vis, [src_quad.astype(np.int32)], True, (0, 255, 255), 2)
if vp is not None:
    vx, vy = int(vp[0]), int(vp[1])
    if 0 <= vx < w and 0 <= vy < h:
        cv2.circle(trap_vis, (vx, vy), 8, (0, 0, 255), -1)
        cv2.putText(trap_vis, "VP", (vx + 10, vy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 8.2 鸟瞰图 + 检出车道线 + 投影车辆位置
bev_vis = img_bev.copy()
for px in peaks:
    cv2.line(bev_vis, (px, 0), (px, BEV_H), (0, 255, 255), 2)
for bx in veh_bev_xs:
    bxi = int(bx)
    if 0 <= bxi < BEV_W:
        cv2.circle(bev_vis, (bxi, BEV_H - 30), 10, (0, 255, 0), -1)

# 8.3 列直方图可视化
hist_vis = np.zeros((250, BEV_W, 3), dtype=np.uint8)
if smooth.max() > 0:
    nh_ = (smooth / smooth.max() * 230).astype(np.int32)
    for x in range(BEV_W):
        cv2.line(hist_vis, (x, 250), (x, 250 - int(nh_[x])),
                 (200, 200, 200), 1)
for px in peaks:
    cv2.line(hist_vis, (px, 0), (px, 250), (0, 255, 255), 1)

# 8.4 单独保存
cv2.imwrite(os.path.join(DEBUG_DIR, "01_edges.jpg"),         edges_roi)
cv2.imwrite(os.path.join(DEBUG_DIR, "02_lane_color_mask.jpg"), lane_color_mask)
cv2.imwrite(os.path.join(DEBUG_DIR, "03_vehicle_mask.jpg"),  filled)
cv2.imwrite(os.path.join(DEBUG_DIR, "04_ipm_trapezoid.jpg"), trap_vis)
cv2.imwrite(os.path.join(DEBUG_DIR, "05_bev_with_lanes.jpg"), bev_vis)
cv2.imwrite(os.path.join(DEBUG_DIR, "06_lane_histogram.jpg"), hist_vis)

# 8.5 总览拼图（4 列 × 2 行）
cell_w, cell_h = 480, 320
panels = [
    label(fit(img,                 cell_w, cell_h), "01 Original"),
    label(fit(to_bgr(edges_roi),   cell_w, cell_h), "02 Canny edges"),
    label(fit(to_bgr(lane_color_mask), cell_w, cell_h), "03 White+Yellow mask"),
    label(fit(trap_vis,            cell_w, cell_h), "04 IPM trapezoid + VP"),
    label(fit(bev_vis,             cell_w, cell_h), "05 BEV + lanes + cars"),
    label(fit(hist_vis,            cell_w, cell_h), "06 Column histogram"),
    label(fit(to_bgr(filled),      cell_w, cell_h), "07 Vehicle mask"),
    label(fit(output,              cell_w, cell_h), "08 Final result"),
]
panel = np.zeros((cell_h * 2, cell_w * 4, 3), dtype=np.uint8)
for idx, p in enumerate(panels):
    r, c = idx // 4, idx % 4
    panel[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = p
cv2.imwrite(os.path.join(DEBUG_DIR, "00_panel.jpg"), panel)
print(f"[Debug] 调试输出目录：{DEBUG_DIR}")


# ═════════════════════════════════════════════════════════════
# 9. 显示
# ═════════════════════════════════════════════════════════════
show(output)
