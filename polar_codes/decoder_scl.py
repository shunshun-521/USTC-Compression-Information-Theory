"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
  _prepare_llr,
  _bit_reversed,
  _upper_llr_exact,
  _lower_llr,
  _active_llr_level,
  _active_bit_level,
  f_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
  reg ^= int(bit) << (crc_length - 1)
  if reg & (1 << (crc_length - 1)):
    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
  else:
    reg = (reg << 1) & ((1 << crc_length) - 1)
  return reg


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
  reg = 0
  for bit in info_bits:
    reg = _crc_step(reg, bit, poly, crc_length)
  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 CRC 是否正确"""
  bits = np.asarray(bits, dtype=int)
  poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
  reg = 0
  for bit in bits:
    reg = _crc_step(reg, bit, poly, crc_length)
  return reg == 0


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy 优化）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.frozen_set = set(np.where(self.frozen_bits)[0])
    self.list_size = list_size
    self.crc_length = crc_length

  def _pm_update(self, pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == hard else abs(llr))

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, pm)"""
    llr = _prepare_llr(llr_ch)
    N = self.N
    n = self.n
    L_size = self.list_size

    paths = [{
      'L': np.full((N, n + 1), np.nan, dtype=np.float64),
      'B': np.full((N, n + 1), np.nan),
      'PM': 0.0,
      'u_hat': np.zeros(N, dtype=int),
    }]
    paths[0]['L'][:, 0] = llr

    for phi_natural in range(N):
      l = _bit_reversed(phi_natural, n)
      new_paths = []

      for path in paths:
        L = path['L']
        B = path['B']
        PM = path['PM']
        u_hat = path['u_hat']

        for s in range(n - _active_llr_level(l, n), n):
          block = 2 ** (s + 1)
          branch = block // 2
          if block > N:
            continue
          for j in range(l, N, block):
            if j % block < branch:
              L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch, s])
            else:
              L[j, s + 1] = _lower_llr(L[j, s], L[j - branch, s], int(B[j - branch, s + 1]))

        llr_bit = L[l, n]
        candidates = [(0, self._pm_update(PM, llr_bit, 0))] if l in self.frozen_set else [
          (0, self._pm_update(PM, llr_bit, 0)),
          (1, self._pm_update(PM, llr_bit, 1)),
        ]

        for u, new_pm in candidates:
          new_path = {
            'L': L.copy(),
            'B': B.copy(),
            'PM': new_pm,
            'u_hat': u_hat.copy(),
          }
          new_path['B'][l, n] = u
          new_path['u_hat'][l] = u

          if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
              block = 2 ** s
              branch = block // 2
              for j in range(l, -1, -block):
                if j % block >= branch:
                  new_path['B'][j - branch, s - 1] = int(new_path['B'][j, s]) ^ int(
                    new_path['B'][j - branch, s]
                  )
                  new_path['B'][j, s - 1] = new_path['B'][j, s]

          new_paths.append(new_path)

      new_paths.sort(key=lambda p: p['PM'])
      paths = new_paths[:L_size]

    best_path = None
    if self.crc_length > 0:
      crc_paths = []
      for path in paths:
        info_bits = path['u_hat'][~self.frozen_bits]
        if crc_check(info_bits, self.crc_length):
          crc_paths.append(path)
      if crc_paths:
        best_path = min(crc_paths, key=lambda p: p['PM'])

    if best_path is None:
      best_path = min(paths, key=lambda p: p['PM'])

    return best_path['u_hat'], best_path['PM']
