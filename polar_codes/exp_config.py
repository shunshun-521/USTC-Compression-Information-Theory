"""快速实验配置：设置环境变量 POLAR_QUICK=1 启用缩减仿真规模。"""
import os

_QUICK = os.environ.get('POLAR_QUICK', '0') == '1'

if _QUICK:
    MAX_FRAMES = 80
    MIN_ERRORS = 5
    EB_N0_RANGE_EXP1 = [2.5, 3.5, 4.0]
    EB_N0_RANGE_EXP23 = [2.5, 3.5, 4.5]
    N_LIST_EXP1 = [256, 512, 1024]
    N_LIST_EXP3 = [256, 512]
    L_LIST_EXP2 = [2, 4]
    CASCL_LIST_SIZE = 4
else:
    MAX_FRAMES = 100000
    MIN_ERRORS = 100
    EB_N0_RANGE_EXP1 = None
    EB_N0_RANGE_EXP23 = None
    N_LIST_EXP1 = [256, 512, 1024]
    N_LIST_EXP3 = [256, 512]
    L_LIST_EXP2 = [2, 4, 8]
    CASCL_LIST_SIZE = 8
