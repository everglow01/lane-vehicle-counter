"""车辆分割与 bbox 过滤。"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class VehicleResult:
    vehicles: list
    vehicle_mask: np.ndarray


def detect_vehicles(img, lane_result):
    h, w = img.shape[:2]
    H_ch = lane_result.H_ch
    S_ch = lane_result.S_ch
    V_ch = lane_result.V_ch
    blur = lane_result.blur
    white_mask = lane_result.white_mask
    yellow_mask = lane_result.yellow_mask
    lane_color_mask = lane_result.lane_color_mask

    # 车辆 ROI：聚焦右侧主道——左边界绕开中央隔离带，右边界绕开高墙
    veh_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(veh_roi, [np.array([
        [int(w * 0.28), h],
        [int(w * 0.36), int(h * 0.02)],
        [int(w * 0.66), int(h * 0.02)],
        [int(w * 0.78), h],
    ], dtype=np.int32)], (255,))

    median_band = _build_median_band(yellow_mask, veh_roi, h)
    veh_roi = cv2.bitwise_and(veh_roi, cv2.bitwise_not(median_band))

    v_road = V_ch[veh_roi > 0]
    v_th = int(min(np.percentile(v_road, 14), 115))
    v_th = max(v_th, 60)
    dark_mask = ((V_ch < v_th) & (veh_roi > 0)).astype(np.uint8) * 255

    color_mask = ((S_ch > 60) & (veh_roi > 0)).astype(np.uint8) * 255
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(yellow_mask))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))

    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    edge_strong = (sobel_mag > 35).astype(np.uint8) * 255
    edge_density = cv2.dilate(edge_strong,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                              iterations=2)

    dark_mask = cv2.bitwise_and(dark_mask, edge_density)
    color_mask = cv2.bitwise_and(color_mask, edge_density)

    dash_corridor = np.zeros((h, w), dtype=np.uint8)
    for (p1, p2) in lane_result.lane_boundaries_orig:
        cv2.line(dash_corridor, p1, p2, (255,), 20, cv2.LINE_AA)

    bright_mask = ((V_ch > 210) & (S_ch < 60) & (veh_roi > 0)).astype(np.uint8) * 255
    bright_mask = cv2.bitwise_and(bright_mask, cv2.bitwise_not(dash_corridor))
    bright_mask = cv2.bitwise_and(bright_mask, edge_density)

    far_gray_mask = _detect_far_gray_vehicles(
        S_ch, V_ch, veh_roi, lane_color_mask, edge_density, h, w
    )

    vehicle_mask = cv2.bitwise_or(dark_mask, color_mask)
    vehicle_mask = cv2.bitwise_or(vehicle_mask, bright_mask)
    vehicle_mask = cv2.bitwise_or(vehicle_mask, far_gray_mask)

    filled = _fill_vehicle_mask(vehicle_mask)
    vehicles = _filter_vehicle_components(filled, H_ch, S_ch, h, w)

    print(f"[Vehicle] 检出车辆数：{len(vehicles)}")
    return VehicleResult(vehicles=vehicles, vehicle_mask=filled)


def _build_median_band(yellow_mask, veh_roi, h):
    median_seed = cv2.dilate(yellow_mask,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)),
                             iterations=2)
    median_band = np.zeros_like(veh_roi)
    cnts_y, _ = cv2.findContours(median_seed, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts_y:
        if cv2.contourArea(c) < 4000:
            continue
        cv2.drawContours(median_band, [c], -1, (255,), -1)

        [vx_, vy_, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        if abs(vy_) < 1e-3:
            continue
        pts = c.reshape(-1, 2)
        t = (pts[:, 1] - y0) / vy_
        t_min, t_max = float(t.min()), float(t.max())
        span = max(abs(t_max - t_min), 200)
        if vy_ < 0:
            t_top = t_max + span
            t_bot = t_min
        else:
            t_top = t_min - span
            t_bot = t_max
        p_top = (int(x0 + vx_ * t_top), int(y0 + vy_ * t_top))
        p_bot = (int(x0 + vx_ * t_bot), int(y0 + vy_ * t_bot))
        cv2.line(median_band, p_top, p_bot, (255,), 60, cv2.LINE_AA)
    return median_band


def _detect_far_gray_vehicles(S_ch, V_ch, veh_roi, lane_color_mask, edge_density, h, w):
    far_roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(far_roi, [np.array([
        [int(w * 0.34), int(h * 0.01)],
        [int(w * 0.68), int(h * 0.01)],
        [int(w * 0.72), int(h * 0.22)],
        [int(w * 0.30), int(h * 0.22)],
    ], dtype=np.int32)], (255,))
    local_bg = cv2.GaussianBlur(V_ch, (31, 31), 0)
    local_contrast = (cv2.absdiff(V_ch, local_bg) > 18).astype(np.uint8) * 255
    lane_corridor = cv2.dilate(lane_color_mask,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                               iterations=1)
    far_gray_mask = ((S_ch < 80) & (V_ch > 95) & (V_ch < 215) &
                     (veh_roi > 0) & (far_roi > 0)).astype(np.uint8) * 255
    far_gray_mask = cv2.bitwise_and(far_gray_mask, cv2.bitwise_not(lane_corridor))
    far_gray_mask = cv2.bitwise_and(far_gray_mask, edge_density)
    far_gray_mask = cv2.bitwise_and(far_gray_mask, local_contrast)
    return far_gray_mask


def _fill_vehicle_mask(vehicle_mask):
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_OPEN, k3, iterations=1)
    vehicle_mask = cv2.morphologyEx(vehicle_mask, cv2.MORPH_CLOSE, k5, iterations=2)

    contours_raw, _ = cv2.findContours(vehicle_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(vehicle_mask)
    for cnt in contours_raw:
        a = cv2.contourArea(cnt)
        if a < 50:
            continue
        if a > 6000:
            cv2.drawContours(filled, [cnt], -1, (255,), -1)
        else:
            cv2.drawContours(filled, [cv2.convexHull(cnt)], -1, (255,), -1)
    return filled


def _filter_vehicle_components(filled, H_ch, S_ch, h, w):
    n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)

    vehicles = []
    for i in range(1, n_comp):
        x, y, bw, bh, area = stats[i]
        if bw == 0 or bh == 0:
            continue
        aspect = bw / float(bh)
        rel_y = (y + bh / 2.0) / h
        min_area = 40 + int(1500 * rel_y)
        max_area = 3000 + int(70000 * rel_y)
        if not (min_area < area < max_area):
            continue
        # 贴近顶部的远车常被画面上边缘截断，只剩较扁的车尾/车顶。
        max_aspect = 4.0 if rel_y < 0.05 and area < 400 else 3.0
        if not (0.4 < aspect < max_aspect):
            continue
        if bw > w * 0.45 or bh > h * 0.55:
            continue
        if area / float(bw * bh) < 0.30:
            continue

        blob = (labels == i)
        h_in, s_in = H_ch[blob], S_ch[blob]
        yellow_ratio = float(((h_in >= 15) & (h_in <= 35) & (s_in > 80)).mean())
        if yellow_ratio > 0.40:
            continue

        vehicles.append((int(x), int(y), int(bw), int(bh)))
    return vehicles

