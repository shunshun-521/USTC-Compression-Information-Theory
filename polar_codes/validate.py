"""极化码模块数值校验（编码、SC/SCL 一致性、高信噪比无损译码）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from run_exp1 import run_unit_tests

if __name__ == "__main__":
    run_unit_tests()
    print("validate.py: 全部通过")
