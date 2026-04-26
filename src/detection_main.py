"""
高速公路车辆与车道检测主流程
─────────────────────────────────────────────────────────────
整体流程：
  1. 读取图像
  2. 检测车道与 IPM 透视信息
  3. 检测车辆 bbox
  4. 统计每个车道车辆数
  5. 绘制结果并保存调试图
"""

import os

import cv2

from config import BEV_H, BEV_W, DEBUG_DIR, IMG_PATH, OUT_PATH
from lane_detection import count_vehicles_by_lane, detect_lanes
from vehicle_detection import detect_vehicles
from visualization import draw_final_result, save_debug_images, show


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)

    img = cv2.imread(IMG_PATH)
    if img is None:
        raise FileNotFoundError(f"无法读取图像：{IMG_PATH}")

    lane_result = detect_lanes(img, BEV_W, BEV_H)
    vehicle_result = detect_vehicles(img, lane_result)
    all_lane_counts, veh_bev_xs = count_vehicles_by_lane(
        vehicle_result.vehicles, lane_result, BEV_W
    )

    output = draw_final_result(img, lane_result, vehicle_result, all_lane_counts)
    cv2.imwrite(OUT_PATH, output)
    print(f"[Output] 结果已保存：{OUT_PATH}")

    save_debug_images(
        img, output, lane_result, vehicle_result, veh_bev_xs,
        DEBUG_DIR, BEV_W, BEV_H
    )
    show(output)


if __name__ == "__main__":
    main()

