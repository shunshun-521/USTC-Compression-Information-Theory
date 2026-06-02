"""仿真参数（支持环境变量加速调试）"""
import os

QUICK = os.environ.get('POLAR_QUICK', '0') == '1'

MAX_FRAMES = int(os.environ.get(
    'POLAR_MAX_FRAMES', '2000' if QUICK else '100000'
))
MIN_ERRORS = int(os.environ.get(
    'POLAR_MIN_ERRORS', '20' if QUICK else '100'
))
