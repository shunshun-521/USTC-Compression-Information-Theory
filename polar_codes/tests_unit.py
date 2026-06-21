"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, polar_encode_matrix
from simulation import run_unit_tests


def test_crc():
  info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  coded[-1] ^= 1
  assert not crc_check(coded, 8)


def test_ga_n8():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  assert len(info) == 4
  assert len(frozen) == 4
  assert len(set(info) & set(frozen)) == 0


if __name__ == "__main__":
  run_unit_tests()
  test_crc()
  test_ga_n8()
  print("tests_unit.py: 全部通过")
