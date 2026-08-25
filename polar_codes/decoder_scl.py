"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _permute_llr
from encoder import bit_reversal_permutation


CRC_POLYS = {
  8: 0x07,
  16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后。"""
  info_bits = np.asarray(info_bits, dtype=np.int8)
  poly = CRC_POLYS[crc_length]
  mask = (1 << crc_length) - 1
  reg = 0
  for bit in info_bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask

  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
    dtype=np.int8,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  if crc_length == 0:
    return True
  poly = CRC_POLYS[crc_length]
  mask = (1 << crc_length) - 1
  reg = 0
  for bit in bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
      reg = ((reg << 1) ^ poly) & mask
    else:
      reg = (reg << 1) & mask
  return reg == 0


class SCLDecoder:
  """SCL 译码器（Lazy Copy 风格的路径列表管理）。"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]
    self.metrics = [0.0]
    self.decisions = [np.zeros(N, dtype=np.int8)]

  def _penalty(self, llr, bit):
    return 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)

  def _leaf(self, llrs, index):
    if self.frozen_bits[index]:
      for path, llr in enumerate(llrs):
        self.metrics[path] += self._penalty(float(llr[0]), 0)
        self.decisions[path][index] = 0
      return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

    candidates = []
    for path, llr in enumerate(llrs):
      for bit in (0, 1):
        candidates.append(
          (self.metrics[path] + self._penalty(float(llr[0]), bit), path, bit)
        )
    candidates.sort(key=lambda item: item[0])
    kept = candidates[: self.list_size]

    new_metrics = []
    new_decisions = []
    betas = []
    parent_map = []
    for metric, parent, bit in kept:
      new_metrics.append(metric)
      decision = self.decisions[parent].copy()
      decision[index] = bit
      new_decisions.append(decision)
      betas.append(np.array([bit], dtype=np.int8))
      parent_map.append(parent)

    self.metrics = new_metrics
    self.decisions = new_decisions
    return betas, parent_map

  def _node(self, llrs, base, length):
    if length == 1:
      return self._leaf(llrs, base)

    half = length // 2
    upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
    beta_upper, map_upper = self._node(upper, base, half)

    lower = [
      g_operation(
        llrs[map_upper[p]][:half],
        llrs[map_upper[p]][half:],
        beta_upper[p],
      )
      for p in range(len(map_upper))
    ]
    beta_lower, map_lower = self._node(lower, base + half, half)

    beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
    betas = [
      np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
      for p in range(len(beta_lower))
    ]
    parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
    return betas, parent_map

  def decode(self, llr_ch):
    llr_ch = _permute_llr(llr_ch, self.N)
    self.metrics = [0.0]
    self.decisions = [np.zeros(self.N, dtype=np.int8)]
    self._node([llr_ch], 0, self.N)

    best_idx = 0
    if self.crc_length > 0:
      valid = []
      for idx, decision in enumerate(self.decisions):
        info_bits = decision[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          valid.append(idx)
      if valid:
        best_idx = min(valid, key=lambda i: self.metrics[i])
      else:
        best_idx = int(np.argmin(self.metrics))
    else:
      best_idx = int(np.argmin(self.metrics))

    return self.decisions[best_idx].copy(), self.metrics[best_idx]
