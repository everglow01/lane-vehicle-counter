"""项目路径与全局尺寸配置。"""

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

IMG_PATH = os.path.join(PROJECT_ROOT, "image", "cim019.jpg")
OUT_PATH = os.path.join(PROJECT_ROOT, "result.jpg")
DEBUG_DIR = os.path.join(PROJECT_ROOT, "debug")

BEV_W, BEV_H = 600, 800

