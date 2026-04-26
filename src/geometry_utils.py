"""几何、IPM 与一维寻峰工具函数。"""

import cv2
import numpy as np


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


def extend_line_to_image(p1, p2, h, w):
    """把已检测到的车道边界按直线延伸到画面顶/底，仅用于最终绘图。"""
    x1, y1 = p1
    x2, y2 = p2
    if abs(y2 - y1) < 1e-6:
        return p1, p2
    x_top = int(round(x1 + (0 - y1) * (x2 - x1) / (y2 - y1)))
    x_bot = int(round(x1 + ((h - 1) - y1) * (x2 - x1) / (y2 - y1)))
    ok, q1, q2 = cv2.clipLine((0, 0, w, h), (x_top, 0), (x_bot, h - 1))
    return (q1, q2) if ok else (p1, p2)

