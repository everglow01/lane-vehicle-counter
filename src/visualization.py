"""最终结果与调试图绘制。"""

import os

import cv2
import numpy as np

from geometry_utils import extend_line_to_image


def draw_final_result(img, lane_result, vehicle_result, all_lane_counts):
    output = img.copy()
    h, w = img.shape[:2]
    lane_boundaries_draw = [
        extend_line_to_image(p1, p2, h, w)
        for (p1, p2) in lane_result.lane_boundaries_orig
    ]

    for (p1, p2) in lane_boundaries_draw:
        cv2.line(output, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

    busiest_msg = "无"
    if lane_result.num_lanes > 0:
        busiest = int(np.argmax(all_lane_counts))
        p1, p2 = lane_boundaries_draw[busiest]
        p3, p4 = lane_boundaries_draw[busiest + 1]
        cv2.line(output, p1, p2, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.line(output, p3, p4, (0, 0, 255), 4, cv2.LINE_AA)
        busiest_msg = f"车道 {busiest + 1}（{all_lane_counts[busiest]} 辆）"
        print(f"[Busiest] {busiest_msg}")

    for (x, y, bw, bh) in vehicle_result.vehicles:
        cv2.rectangle(output, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    cv2.putText(output, f"Vehicles: {len(vehicle_result.vehicles)}  Lanes: {lane_result.num_lanes}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2,
                cv2.LINE_AA)
    lane_count_text = "Lane counts: " + "  ".join(
        f"{li + 1}:{cnt}" for li, cnt in enumerate(all_lane_counts)
    )
    cv2.putText(output, lane_count_text, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2, cv2.LINE_AA)
    return output


def save_debug_images(img, output, lane_result, vehicle_result, veh_bev_xs,
                      debug_dir, bev_w, bev_h):
    trap_vis = img.copy()
    h, w = img.shape[:2]
    cv2.polylines(trap_vis, [lane_result.src_quad.astype(np.int32)], True,
                  (0, 255, 255), 2)
    if lane_result.vp is not None:
        vx, vy = int(lane_result.vp[0]), int(lane_result.vp[1])
        if 0 <= vx < w and 0 <= vy < h:
            cv2.circle(trap_vis, (vx, vy), 8, (0, 0, 255), -1)
            cv2.putText(trap_vis, "VP", (vx + 10, vy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    bev_vis = lane_result.img_bev.copy()
    for px in lane_result.peaks:
        cv2.line(bev_vis, (px, 0), (px, bev_h), (0, 255, 255), 2)
    for bx in veh_bev_xs:
        bxi = int(bx)
        if 0 <= bxi < bev_w:
            cv2.circle(bev_vis, (bxi, bev_h - 30), 10, (0, 255, 0), -1)

    hist_vis = np.zeros((250, bev_w, 3), dtype=np.uint8)
    if lane_result.smooth.max() > 0:
        nh_ = (lane_result.smooth / lane_result.smooth.max() * 230).astype(np.int32)
        for x in range(bev_w):
            cv2.line(hist_vis, (x, 250), (x, 250 - int(nh_[x])),
                     (200, 200, 200), 1)
    for px in lane_result.peaks:
        cv2.line(hist_vis, (px, 0), (px, 250), (0, 255, 255), 1)

    cv2.imwrite(os.path.join(debug_dir, "01_edges.jpg"), lane_result.edges_roi)
    cv2.imwrite(os.path.join(debug_dir, "02_lane_color_mask.jpg"),
                lane_result.lane_color_mask)
    cv2.imwrite(os.path.join(debug_dir, "03_vehicle_mask.jpg"),
                vehicle_result.vehicle_mask)
    cv2.imwrite(os.path.join(debug_dir, "04_ipm_trapezoid.jpg"), trap_vis)
    cv2.imwrite(os.path.join(debug_dir, "05_bev_with_lanes.jpg"), bev_vis)
    cv2.imwrite(os.path.join(debug_dir, "06_lane_histogram.jpg"), hist_vis)

    cell_w, cell_h = 480, 320
    panels = [
        label(fit(img, cell_w, cell_h), "01 Original"),
        label(fit(to_bgr(lane_result.edges_roi), cell_w, cell_h), "02 Canny edges"),
        label(fit(to_bgr(lane_result.lane_color_mask), cell_w, cell_h),
              "03 White+Yellow mask"),
        label(fit(trap_vis, cell_w, cell_h), "04 IPM trapezoid + VP"),
        label(fit(bev_vis, cell_w, cell_h), "05 BEV + lanes + cars"),
        label(fit(hist_vis, cell_w, cell_h), "06 Column histogram"),
        label(fit(to_bgr(vehicle_result.vehicle_mask), cell_w, cell_h),
              "07 Vehicle mask"),
        label(fit(output, cell_w, cell_h), "08 Final result"),
    ]
    panel = np.zeros((cell_h * 2, cell_w * 4, 3), dtype=np.uint8)
    for idx, p in enumerate(panels):
        r, c = idx // 4, idx % 4
        panel[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = p
    cv2.imwrite(os.path.join(debug_dir, "00_panel.jpg"), panel)
    print(f"[Debug] 调试输出目录：{debug_dir}")


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
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

