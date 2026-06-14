"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import copy

from encoder import bit_reversal_permutation
from sc_core import (
    _frozen_to_info_set,
    _init_matrices,
    _sc_step_to,
    _path_metric_update,
    _sc_decode_core,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    使用标准多项式：CRC-8 (0x07), CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    msg = bits[:-crc_length]
    encoded = crc_encode(msg, crc_length)
    return np.array_equal(encoded, bits)


def _scl_decode_core(y_llr, info_set, list_size, crc_length=0):
  N = y_llr.size
  n = int(np.log2(N))
  llr0, bit0, _ = _init_matrices(N, y_llr)
  split_pos = list(info_set)

  llr_list = [llr0]
  bit_list = [bit0]
  pm_list = [0.0]
  split_loc = 0

  while split_loc < len(split_pos):
    pos = split_pos[split_loc]
    prev = split_pos[split_loc - 1] if split_loc > 0 else -1
    l_now = len(pm_list)
    new_llr, new_bit, new_pm = [], [], []

    for i in range(l_now):
      llr_m = copy.deepcopy(llr_list[i])
      bit_m = copy.deepcopy(bit_list[i])
      pm0 = pm_list[i]
      llr_m, bit_m = _sc_step_to(llr_m, bit_m, info_set, pos)
      seg_llr = llr_m[n][prev + 1:pos + 1]
      seg_bit = bit_m[n][prev + 1:pos + 1]
      pm_good = pm0 + _path_metric_update(seg_llr, seg_bit)

      new_llr.append(llr_m)
      new_bit.append(bit_m)
      new_pm.append(pm_good)

      bit_bad = copy.deepcopy(bit_m)
      bit_bad[n][pos] = 1 - bit_bad[n][pos]
      seg_bit_bad = bit_bad[n][prev + 1:pos + 1]
      pm_bad = pm0 + _path_metric_update(seg_llr, seg_bit_bad)
      new_llr.append(llr_m)
      new_bit.append(bit_bad)
      new_pm.append(pm_bad)

    order = np.argsort(new_pm)
    keep = order[:list_size]
    llr_list = [new_llr[i] for i in keep]
    bit_list = [new_bit[i] for i in keep]
    pm_list = [new_pm[i] for i in keep]
    split_loc += 1

  if split_pos[-1] != N - 1:
    l_now = len(pm_list)
    new_llr, new_bit, new_pm = [], [], []
    for i in range(l_now):
      llr_m = copy.deepcopy(llr_list[i])
      bit_m = copy.deepcopy(bit_list[i])
      pm0 = pm_list[i]
      llr_m, bit_m = _sc_step_to(llr_m, bit_m, info_set, N - 1)
      prev = split_pos[-1]
      seg_llr = llr_m[n][prev + 1:N]
      seg_bit = bit_m[n][prev + 1:N]
      new_llr.append(llr_m)
      new_bit.append(bit_m)
      new_pm.append(pm0 + _path_metric_update(seg_llr, seg_bit))
    order = np.argsort(new_pm)
    keep = order[:list_size]
    bit_list = [new_bit[i] for i in keep]
    pm_list = [new_pm[i] for i in keep]

  order = np.argsort(pm_list)
  best_u = None
  best_pm = None

  if crc_length > 0:
    for idx in order:
      u_cand = bit_list[idx][n].astype(int)
      info_bits = u_cand[list(info_set)]
      if crc_check(info_bits, crc_length):
        return u_cand, pm_list[idx]

  idx = order[0]
  return bit_list[idx][n].astype(int), pm_list[idx]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits)
        self.info_set = _frozen_to_info_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        u_hat, pm = _scl_decode_core(
            llr_ch[self.br],
            self.info_set,
            self.list_size,
            self.crc_length,
        )
        return u_hat.astype(int), pm
