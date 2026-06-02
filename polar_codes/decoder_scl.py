"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    SCDecoderState,
    bit_reversed_index,
    f_operation,
    g_operation,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(1):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) & ((1 << width) - 1)) ^ (poly & ((1 << width) - 1))
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, width)
    crc_bits = np.array([(rem >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, width)
    return rem == 0


# ==================== SCL 译码器 ====================


class Path:
  """单条译码路径（Lazy Copy：共享 LLR/比特数组引用）"""

  __slots__ = ("pm", "B", "active", "parent_map")

  def __init__(self, N, n):
    self.pm = 0.0
    self.B = np.zeros((N, n + 1), dtype=np.int8)
    self.active = True
    self.parent_map = None


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(math.log2(N))
    self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    self.L = list_size
    self.crc_length = crc_length
    self.decode_order = [bit_reversed_index(i, self.n) for i in range(N)]

  def _path_metric_penalty(self, llr, u_bit):
    """与 LLR 符号不一致时加 |LLR|"""
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)

  def _update_llrs_path(self, path, l):
    """对单条路径更新 LLR（复用 SC 逻辑）"""
    state = SCDecoderState(self.N, np.zeros(self.N, dtype=bool))
    state.L = path.L if hasattr(path, "L") else None

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N, n = self.N, self.n

    # 路径列表：每条路径维护 L 矩阵与 B 矩阵
    paths = []
    p0 = {
        "pm": 0.0,
        "L": np.full((N, n + 1), np.nan, dtype=np.float64),
        "B": np.zeros((N, n + 1), dtype=np.int8),
    }
    p0["L"][:, 0] = llr_ch.copy()
    paths.append(p0)

    for phi_natural in range(N):
      l = self.decode_order[phi_natural]
      new_paths = []

      for path in paths:
        self._sc_update_llrs(path, l)
        llr_leaf = path["L"][l, n]
        if l in self.frozen_set:
          u0, u1 = 0, None
          pen0 = self._path_metric_penalty(llr_leaf, 0)
          child = self._clone_path(path)
          child["pm"] += pen0
          child["B"][l, n] = 0
          self._sc_update_bits(child, l)
          new_paths.append(child)
        else:
          for u in (0, 1):
            child = self._clone_path(path)
            child["pm"] += self._path_metric_penalty(llr_leaf, u)
            child["B"][l, n] = u
            self._sc_update_bits(child, l)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p["pm"])
      paths = new_paths[: self.L]

    # 选择最优路径
    if self.crc_length > 0:
      info_natural = sorted(set(range(N)) - self.frozen_set)
      K_info = len(info_natural) - self.crc_length
      valid = []
      for p in paths:
        u_br = p["B"][:, n].astype(int)
        br = np.array(self.decode_order, dtype=int)
        u_hat = np.empty(N, dtype=int)
        u_hat[br] = u_br
        payload = u_hat[info_natural[:K_info]]
        if crc_check(
            np.concatenate([payload, u_hat[info_natural[K_info : K_info + self.crc_length]]]),
            self.crc_length,
        ):
          valid.append(p)
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p["pm"])
    u_br = best["B"][:, n].astype(int)
    br = np.array(self.decode_order, dtype=int)
    u_hat = np.empty(N, dtype=int)
    u_hat[br] = u_br
    return u_hat, best["pm"]

  def _clone_path(self, path):
    """路径分裂：复制 PM、L、B（列表较小时开销可接受）"""
    return {
        "pm": path["pm"],
        "L": path["L"].copy(),
        "B": path["B"].copy(),
    }

  def _sc_update_llrs(self, path, l):
    from decoder_sc import active_llr_level

    L, B = path["L"], path["B"]
    for s in range(self.n - active_llr_level(l, self.n), self.n):
      block_size = 2 ** (s + 1)
      branch_size = block_size // 2
      for j in range(l, self.N, block_size):
        if j % block_size < branch_size:
          L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
        else:
          top_bit = B[j - branch_size, s + 1]
          L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

  def _sc_update_bits(self, path, l):
    from decoder_sc import active_bit_level

    B = path["B"]
    if l < self.N // 2:
      return
    for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
      block_size = 2 ** s
      branch_size = block_size // 2
      for j in range(l, -1, -block_size):
        if j % block_size >= branch_size:
          B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
          B[j, s - 1] = B[j, s]
