"""极化码实验公共配置。"""
import os

import numpy as np


def quick_mode():
  return os.environ.get("POLAR_QUICK", "0") == "1"


def exp1_params():
  if quick_mode():
    return {
      "N_LIST": [256],
      "MAX_FRAMES": 2000,
      "MIN_ERRORS": 20,
      "EB_N0_RANGE": np.arange(1.0, 4.0, 0.5),
    }
  return {
    "N_LIST": [256, 512, 1024],
    "MAX_FRAMES": 100000,
    "MIN_ERRORS": 100,
    "EB_N0_RANGE": np.arange(0.0, 5.5, 0.25),
  }


def exp2_params():
  if quick_mode():
    return {
      "N": 256,
      "L_LIST": [2, 4],
      "MAX_FRAMES": 2000,
      "MIN_ERRORS": 20,
      "EB_N0_RANGE": np.arange(1.5, 4.0, 0.5),
    }
  return {
    "N": 512,
    "L_LIST": [2, 4, 8],
    "MAX_FRAMES": 100000,
    "MIN_ERRORS": 100,
    "EB_N0_RANGE": np.arange(1.0, 5.5, 0.25),
  }


def exp3_params():
  if quick_mode():
    return {
      "N_LIST": [256],
      "MAX_FRAMES": 2000,
      "MIN_ERRORS": 20,
      "EB_N0_RANGE": np.arange(1.5, 4.0, 0.5),
      "MAX_ITER": 50,
    }
  return {
    "N_LIST": [256, 512],
    "MAX_FRAMES": 100000,
    "MIN_ERRORS": 100,
    "EB_N0_RANGE": np.arange(1.0, 5.5, 0.25),
    "MAX_ITER": 50,
  }
