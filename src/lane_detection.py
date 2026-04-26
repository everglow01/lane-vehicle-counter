"""车道线、IPM 与车道车辆计数。"""

from dataclasses import dataclass

import cv2
import numpy as np

from geometry_utils import (
    build_ipm_src,
    find_peaks_1d,
    find_vanishing_point,
    fit_solid_line,
)


@dataclass
class LaneResult:
    gray: np.ndarray
    blur: np.ndarray
    hsv: np.ndarray
    H_ch: np.ndarray
    S_ch: np.ndarray
    V_ch: np.ndarray
    white_mask: np.ndarray
    yellow_mask: np.ndarray
    lane_color_mask: np.ndarray
    edges_roi: np.ndarray
    vp: tuple | None
    src_quad: np.ndarray
    H_mat: np.ndarray
    H_inv: np.ndarray
    img_bev: np.ndarray
    lane_bev: np.ndarray
    peaks: list
    smooth: np.ndarray
    lane_boundaries_bev: list
    lane_boundaries_orig: list
    num_lanes: int


def detect_lanes(img, bev_w, bev_h):
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_ch, S_ch, V_ch = cv2.split(hsv)

    white_mask = ((S_ch < 50) & (V_ch > 180)).astype(np.uint8) * 255
    yellow_mask = ((H_ch >= 15) & (H_ch <= 35) &
                   (S_ch > 80) & (V_ch > 80)).astype(np.uint8) * 255
    lane_color_mask = cv2.bitwise_or(white_mask, yellow_mask)

    edges = cv2.Canny(blur, 50, 150)
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

    src_quad = build_ipm_src(vp, h, w)
    dst_quad = np.array([
        [0,     bev_h],
        [0,     0    ],
        [bev_w, 0    ],
        [bev_w, bev_h],
    ], dtype=np.float32)

    H_mat = cv2.getPerspectiveTransform(src_quad, dst_quad)
    H_inv = np.linalg.inv(H_mat)

    img_bev = cv2.warpPerspective(img, H_mat, (bev_w, bev_h))
    lane_bev = cv2.warpPerspective(lane_color_mask, H_mat, (bev_w, bev_h))

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 51))
    lane_bev_d = cv2.dilate(lane_bev, v_kernel, iterations=1)

    col_hist = np.sum(lane_bev_d > 0, axis=0).astype(np.float32)
    ksz = 21
    smooth = np.convolve(col_hist, np.ones(ksz) / ksz, mode="same")

    peaks = find_peaks_1d(smooth, min_h=bev_h * 0.06, min_dist=50,
                          min_prom=bev_h * 0.04)
    peaks = [p for p in peaks if 10 < p < bev_w - 10]
    print(f"[Lane] BEV 内侧白虚线峰值数：{len(peaks)} → {peaks}")

    yellow_pts = _detect_yellow_outer_line(yellow_mask, h)
    shoulder_pts = _detect_right_shoulder_line(white_mask, vp, h, w)

    bev_x_yellow = _bottom_to_bev_x(yellow_pts, H_mat)
    bev_x_shoulder = _bottom_to_bev_x(shoulder_pts, H_mat)

    boundaries = []
    for px in peaks:
        p_top, p_bot = _bev_x_to_orig_line(px, H_inv, bev_h)
        boundaries.append((float(px), p_top, p_bot, "inner"))
    if yellow_pts is not None and bev_x_yellow is not None:
        boundaries.append((bev_x_yellow, yellow_pts[0], yellow_pts[1], "outer"))
    if shoulder_pts is not None and bev_x_shoulder is not None:
        boundaries.append((bev_x_shoulder, shoulder_pts[0], shoulder_pts[1], "outer"))
    boundaries.sort(key=lambda b: b[0])

    lane_boundaries_bev = [b[0] for b in boundaries]
    lane_boundaries_orig = [(b[1], b[2]) for b in boundaries]
    num_lanes = max(0, len(lane_boundaries_bev) - 1)
    print(f"[Lane] 黄线 bev_x={bev_x_yellow}, 右路肩 bev_x={bev_x_shoulder}")
    print(f"[Lane] 总边界数：{len(lane_boundaries_bev)}  车道数：{num_lanes}")

    return LaneResult(
        gray=gray,
        blur=blur,
        hsv=hsv,
        H_ch=H_ch,
        S_ch=S_ch,
        V_ch=V_ch,
        white_mask=white_mask,
        yellow_mask=yellow_mask,
        lane_color_mask=lane_color_mask,
        edges_roi=edges_roi,
        vp=vp,
        src_quad=src_quad,
        H_mat=H_mat,
        H_inv=H_inv,
        img_bev=img_bev,
        lane_bev=lane_bev,
        peaks=peaks,
        smooth=smooth,
        lane_boundaries_bev=lane_boundaries_bev,
        lane_boundaries_orig=lane_boundaries_orig,
        num_lanes=num_lanes,
    )


def count_vehicles_by_lane(vehicles, lane_result, bev_w):
    lane_counts = {li: 0 for li in range(lane_result.num_lanes)}
    veh_bev_xs = []
    for (x, y, bw, bh) in vehicles:
        bottom_pts = np.array([[
            [x, y + bh],
            [x + bw, y + bh],
            [x + bw / 2.0, y + bh],
        ]], dtype=np.float32)
        bev_pts = cv2.perspectiveTransform(bottom_pts, lane_result.H_mat)[0]
        veh_left = float(min(bev_pts[0][0], bev_pts[1][0]))
        veh_right = float(max(bev_pts[0][0], bev_pts[1][0]))
        bev_x = float(bev_pts[2][0])
        veh_bev_xs.append(bev_x)

        lane_idx = _assign_lane_by_overlap(
            veh_left, veh_right, bev_x, lane_result.lane_boundaries_bev
        )
        if lane_idx is None:
            continue
        lane_counts[lane_idx] += 1

    all_lane_counts = [lane_counts.get(li, 0) for li in range(lane_result.num_lanes)]
    print("[LaneCounts] " + "  ".join(
        f"车道{li + 1}={cnt}辆" for li, cnt in enumerate(all_lane_counts)
    ))
    return all_lane_counts, veh_bev_xs


def _assign_lane_by_overlap(veh_left, veh_right, veh_center, lane_boundaries):
    if len(lane_boundaries) < 2:
        return None

    overlaps = []
    for li in range(len(lane_boundaries) - 1):
        lane_left = lane_boundaries[li]
        lane_right = lane_boundaries[li + 1]
        overlap = max(0.0, min(veh_right, lane_right) - max(veh_left, lane_left))
        overlaps.append(overlap)

    best_overlap = max(overlaps)
    if best_overlap > 0:
        best_lanes = [li for li, overlap in enumerate(overlaps)
                      if abs(overlap - best_overlap) < 1e-6]
        if len(best_lanes) == 1:
            return best_lanes[0]
        return min(best_lanes, key=lambda li: _lane_center_dist(li, veh_center, lane_boundaries))

    lane_centers = [
        (lane_boundaries[li] + lane_boundaries[li + 1]) / 2.0
        for li in range(len(lane_boundaries) - 1)
    ]
    return int(np.argmin([abs(veh_center - cx) for cx in lane_centers]))


def _lane_center_dist(lane_idx, veh_center, lane_boundaries):
    lane_center = (lane_boundaries[lane_idx] + lane_boundaries[lane_idx + 1]) / 2.0
    return abs(veh_center - lane_center)


def _detect_yellow_outer_line(yellow_mask, h):
    yellow_pts = None
    cnts_yo, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if cnts_yo:
        c_yo = max(cnts_yo, key=cv2.contourArea)
        if cv2.contourArea(c_yo) > 800:
            yellow_pts = fit_solid_line(c_yo, h)
    return yellow_pts


def _detect_right_shoulder_line(white_mask, vp, h, w):
    shoulder_pts = None
    if vp is None:
        return shoulder_pts

    vpx, vpy = vp
    cnts_w, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
    best_score, best_c = -1.0, None
    for c in cnts_w:
        x, y, bw, bh = cv2.boundingRect(c)
        if x + bw < 0.55 * w:
            continue
        if y + bh < 0.6 * h:
            continue
        if bh < 150:
            continue
        if bh < 0.7 * bw:
            continue
        [vx_, vy_, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0,
                                         0.01, 0.01).flatten()
        if abs(vy_) < 1e-3:
            continue
        if abs(vy_ * (vpx - x0) - vx_ * (vpy - y0)) > 30:
            continue
        x_h = x0 + vx_ * ((h - y0) / vy_)
        if not (0.6 * w < x_h < 1.3 * w):
            continue
        score = float(bh + 0.3 * x_h)
        if score > best_score:
            best_score, best_c = score, c
    if best_c is not None:
        shoulder_pts = fit_solid_line(best_c, h)
    return shoulder_pts


def _bottom_to_bev_x(pts, H_mat):
    if pts is None:
        return None
    src = np.array([[pts[1]]], dtype=np.float32)
    return float(cv2.perspectiveTransform(src, H_mat)[0][0][0])


def _bev_x_to_orig_line(bev_x, H_inv, bev_h):
    pts_bev = np.array([[[bev_x, 0]], [[bev_x, bev_h]]], dtype=np.float32)
    pts_orig = cv2.perspectiveTransform(pts_bev, H_inv)
    return tuple(pts_orig[0][0].astype(int)), tuple(pts_orig[1][0].astype(int))
