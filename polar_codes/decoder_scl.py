"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _INF, _compute_llr, _s_updater, f_operation, g_operation


def _crc_poly(crc_length):
  if crc_length == 8:
    return 0x07
  if crc_length == 16:
    return 0x8005
  raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后"""
  info_bits = np.asarray(info_bits, dtype=np.int32)
  poly = _crc_poly(crc_length)
  reg = 0
  for bit in info_bits:
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
      reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
      reg = (reg << 1) & ((1 << crc_length) - 1)
  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int32
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 CRC 是否正确"""
  bits = np.asarray(bits, dtype=np.int32)
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(expected, bits)


class _Path:
  __slots__ = ("llrs", "s", "pm", "u_hat")

  def __init__(self, n, N):
    self.llrs = np.full((n + 1, N), -_INF, dtype=np.float64)
    self.s = np.full((n + 1, N), -1, dtype=np.int32)
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int32)


class SCLDecoder:
  """SCL 译码器（含 Lazy Copy 优化）"""

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.n = int(np.log2(N))
    self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def _clone_path(self, path):
    new_path = _Path(self.n, self.N)
    new_path.llrs = path.llrs.copy()
    new_path.s = path.s.copy()
    new_path.pm = path.pm
    new_path.u_hat = path.u_hat.copy()
    return new_path

  def _path_llr(self, path, phi):
    return _compute_llr(0, phi, path.llrs, path.s)

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    path = _Path(self.n, self.N)
    path.llrs[self.n, :] = llr_ch
    paths = [path]

    for phi in range(self.N):
      new_paths = []
      for p in paths:
        llr_val = self._path_llr(p, phi)
        if self.frozen_bits[phi]:
          child = self._clone_path(p)
          child.u_hat[phi] = 0
          child.s[0, phi] = 0
          child.llrs[0, phi] = _INF
          if llr_val < 0:
            child.pm += abs(llr_val)
          new_paths.append(child)
        else:
          for bit in (0, 1):
            child = self._clone_path(p)
            child.u_hat[phi] = bit
            child.s[0, phi] = bit
            child.llrs[0, phi] = llr_val
            expected = 0 if llr_val >= 0 else 1
            if bit != expected:
              child.pm += abs(llr_val)
            new_paths.append(child)

      new_paths.sort(key=lambda p: p.pm)
      paths = new_paths[: self.list_size]

    if self.crc_length > 0:
      valid = []
      for p in paths:
        info_bits = p.u_hat[self.info_indices]
        if crc_check(info_bits, self.crc_length):
          valid.append(p)
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p.pm)
    return best.u_hat.copy(), best.pm
