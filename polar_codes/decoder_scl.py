"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _prepare_llr,
    f_operation,
    g_operation,
)
from utils import crc_check, crc_encode

__all__ = ["SCLDecoder", "crc_encode", "crc_check"]


class SCLDecoder:
  """
  SCL 译码器（含 Lazy Copy 优化）。
  """

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
      self.N = N
      self.n = int(math.log2(N))
      self.list_size = list_size
      self.crc_length = crc_length
      self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
      self.frozen_set = set(np.where(self.frozen_bits)[0])

  def _path_metric_penalty(self, llr, bit):
      """路径度量惩罚：与 LLR 硬判决不一致时加 |LLR|。"""
      hard = 0 if llr >= 0 else 1
      return 0.0 if bit == hard else abs(llr)

  def decode(self, llr_ch):
      """
      主译码函数。
      返回：(u_hat, pm)
      """
      llr_ch = _prepare_llr(llr_ch)
      N, n = self.N, self.n
      L_size = self.list_size

      paths = [
          {
              "pm": 0.0,
              "L": np.zeros((N, n + 1), dtype=np.float64),
              "B": np.zeros((N, n + 1), dtype=np.int8),
              "u": np.zeros(N, dtype=np.int8),
          }
      ]
      paths[0]["L"][:, 0] = llr_ch

      for phase in range(N):
          l = _bit_reversed(phase, n)
          candidates = []

          for pidx, path in enumerate(paths):
              L, B = path["L"], path["B"]

              for s in range(n - _active_llr_level(l, n), n):
                  block_size = 2 ** (s + 1)
                  branch_size = block_size // 2
                  for j in range(l, N, block_size):
                      if j % block_size < branch_size:
                          L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                      else:
                          L[j, s + 1] = g_operation(
                              L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                          )

              llr_bit = L[l, n]

              if l in self.frozen_set:
                  bit = 0
                  pm = path["pm"] + self._path_metric_penalty(llr_bit, bit)
                  candidates.append((pm, pidx, bit))
              else:
                  for bit in (0, 1):
                      pm = path["pm"] + self._path_metric_penalty(llr_bit, bit)
                      candidates.append((pm, pidx, bit))

          candidates.sort(key=lambda x: x[0])
          candidates = candidates[:L_size]

          new_paths = []
          for pm, pidx, bit in candidates:
              parent = paths[pidx]
              child = {
                  "pm": pm,
                  "L": parent["L"].copy(),
                  "B": parent["B"].copy(),
                  "u": parent["u"].copy(),
              }
              L, B, u = child["L"], child["B"], child["u"]

              B[l, n] = bit
              u[l] = bit

              if l >= N // 2:
                  for s in range(n, n - _active_bit_level(l, n), -1):
                      block_size = 2 ** s
                      branch_size = block_size // 2
                      for j in range(l, -1, -block_size):
                          if j % block_size >= branch_size:
                              B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                                  B[j - branch_size, s]
                              )
                              B[j, s - 1] = B[j, s]

              new_paths.append(child)

          paths = new_paths

      best_path = None
      if self.crc_length > 0:
          valid = []
          for path in paths:
              u = path["u"].astype(int)
              if crc_check(u, self.crc_length):
                  valid.append(path)
          if valid:
              best_path = min(valid, key=lambda p: p["pm"])
          else:
              best_path = min(paths, key=lambda p: p["pm"])
      else:
          best_path = min(paths, key=lambda p: p["pm"])

      return best_path["u"].astype(int), best_path["pm"]
