"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation

# ==================== CRC 工具 ====================

_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  CRC-8: 0x07; CRC-16: 0x8005
  """
  info_bits = np.asarray(info_bits, dtype=int).ravel()
  poly = _CRC_POLY[crc_length]
  reg = np.zeros(crc_length, dtype=int)
  for bit in info_bits:
    fb = bit ^ reg[-1]
    reg[1:] = reg[:-1]
    reg[0] = 0
    if fb:
      for i in range(crc_length):
        if (poly >> i) & 1:
          reg[i] ^= fb
  return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
  """检验 bits 末尾 CRC 是否正确"""
  bits = np.asarray(bits, dtype=int).ravel()
  if len(bits) < crc_length:
    return False
  poly = _CRC_POLY[crc_length]
  reg = np.zeros(crc_length, dtype=int)
  for bit in bits:
    fb = bit ^ reg[-1]
    reg[1:] = reg[:-1]
    reg[0] = 0
    if fb:
      for i in range(crc_length):
        if (poly >> i) & 1:
          reg[i] ^= fb
  return np.all(reg == 0)


# ==================== SCL 辅助 ====================


def compute_llr_at_bit(llr, u_prefix, target_phi, offset=0):
  """
  给定已判决前缀 u_prefix（长度 target_phi），计算第 target_phi 位的叶节点 LLR。
  偶/奇分裂极化树。
  """
  N = len(llr)
  if N == 1:
    return float(llr[0])

  half = N // 2
  if target_phi < offset + half:
    return compute_llr_at_bit(
      f_operation(llr[::2], llr[1::2]),
      u_prefix[: max(0, target_phi - offset)],
      target_phi,
      offset,
    )

  u_left = u_prefix[:half]
  right_llr = g_operation(
    f_operation(u_left.astype(np.float64), llr[::2]),
    llr[1::2],
    u_left,
  )
  return compute_llr_at_bit(
    right_llr,
    u_prefix[half:],
    target_phi,
    offset + half,
  )


def path_metric_update(pm, llr, u):
  hard = 0 if llr >= 0 else 1
  if u != hard:
    pm += abs(llr)
  return pm


# ==================== SCL 译码器 ====================


class SCLDecoder:
  """
  SCL 译码器（Lazy Copy：路径仅在分裂时复制 u_hat 数组）。
  frozen_bits: True 表示冻结位
  """

  def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
    self.N = N
    self.frozen_bits = np.asarray(frozen_bits).astype(bool)
    self.list_size = list_size
    self.crc_length = crc_length
    self.info_indices = np.where(~self.frozen_bits)[0]

  def decode(self, llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    paths = [{"u_hat": np.zeros(self.N, dtype=int), "pm": 0.0}]

    for phi in range(self.N):
      new_paths = []
      for path in paths:
        llr_phi = compute_llr_at_bit(llr_ch, path["u_hat"][:phi], phi)

        if self.frozen_bits[phi]:
          path["pm"] = path_metric_update(path["pm"], llr_phi, 0)
          path["u_hat"][phi] = 0
          new_paths.append(path)
        else:
          for u in (0, 1):
            p = {
              "u_hat": path["u_hat"].copy(),
              "pm": path_metric_update(path["pm"], llr_phi, u),
            }
            p["u_hat"][phi] = u
            new_paths.append(p)

      new_paths.sort(key=lambda x: x["pm"])
      paths = new_paths[: self.list_size]

    return self._select_best(paths)

  def _select_best(self, paths):
    if not paths:
      return np.zeros(self.N, dtype=int), 0.0

    if self.crc_length > 0:
      K_info = len(self.info_indices) - self.crc_length
      info_pos = self.info_indices[:K_info]
      valid = [p for p in paths if crc_check(p["u_hat"][info_pos], self.crc_length)]
      if valid:
        paths = valid

    best = min(paths, key=lambda p: p["pm"])
    return best["u_hat"].copy(), float(best["pm"])
