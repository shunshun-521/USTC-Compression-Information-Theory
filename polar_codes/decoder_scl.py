"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
  _active_bit_level,
  _active_llr_level,
  _bit_reversed_index,
  _update_bits,
  _update_llrs,
  f_operation,
  g_operation,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_division(info_bits, poly, crc_length):
  reg = 0
  for bit in info_bits:
    reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
    if reg & (1 << (crc_length - 1)):
      reg ^= poly
  return np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=int,
  )


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=int)
  poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
  crc_bits = _crc_division(info_bits, poly, crc_length)
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  bits = np.asarray(bits, dtype=int)
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy 优化）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.rev = bit_reversal_permutation(N)
    self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

  @staticmethod
  def _path_metric_update(pm, llr, u):
    expected = 0 if llr >= 0 else 1
    return pm + (0.0 if u == expected else abs(llr))

  def decode(self, llr_ch):
    """主译码函数，返回 (u_hat, pm)。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N, n = self.N, self.n

    paths = [
      {
        "pm": 0.0,
        "L": np.full((N, n + 1), np.nan, dtype=np.float64),
        "B": np.full((N, n + 1), np.nan),
      }
    ]
    paths[0]["L"][:, 0] = llr_ch[self.rev]

    for l in self.decode_order:
      new_paths = []
      for path in paths:
        L, B = path["L"], path["B"]
        _update_llrs(L, B, l, n)
        llr_bit = L[l, n]

        if self.frozen_bits[l]:
          pm = self._path_metric_update(path["pm"], llr_bit, 0)
          child = {
            "pm": pm,
            "L": L.copy(),
            "B": B.copy(),
          }
          child["B"][l, n] = 0
          _update_bits(child["B"], l, n, N)
          new_paths.append(child)
        else:
          for u_val in (0, 1):
            pm = self._path_metric_update(path["pm"], llr_bit, u_val)
            child = {
              "pm": pm,
              "L": L.copy(),
              "B": B.copy(),
            }
            child["B"][l, n] = u_val
            _update_bits(child["B"], l, n, N)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p["pm"])
      paths = new_paths[: self.list_size]

    crc_valid = []
    for path in paths:
      u_hat = path["B"][:, n].astype(int)
      if self.crc_length > 0:
        info_bits = u_hat[~self.frozen_bits]
        crc_valid.append(crc_check(info_bits, self.crc_length))
      else:
        crc_valid.append(True)

    valid_paths = [p for p, ok in zip(paths, crc_valid) if ok]
    best = min(valid_paths if valid_paths else paths, key=lambda p: p["pm"])
    return best["B"][:, n].astype(int), best["pm"]
