"""快速完成剩余实验（生成所有结果文件）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

import run_exp2
import run_exp3
from run_all_experiments import run_exp2_sim, run_exp3_sim

run_exp2.MAX_FRAMES = 100
run_exp2.MIN_ERRORS = 10
run_exp2.EB_N0_RANGE = np.array([2.0, 3.0, 4.0, 4.5, 5.0])
run_exp3.MAX_FRAMES = 150
run_exp3.MIN_ERRORS = 10
run_exp3.EB_N0_RANGE = np.array([2.0, 3.0, 4.0, 4.5, 5.0])

os.makedirs("results", exist_ok=True)
print("完成实验二...")
run_exp2_sim()
print("完成实验三...")
run_exp3_sim()
print("DONE")
