"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _SCDState,
)


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  CRC-8: 0x07 (x^8 + x^2 + x + 1)
  CRC-16: 0x8005
  """
  info_bits = np.asarray(info_bits, dtype=int)
  if crc_length == 8:
    poly = 0x07
  elif crc_length == 16:
    poly = 0x8005
  else:
    raise ValueError("crc_length must be 8 or 16")

  mask = (1 << crc_length) - 1
  reg = 0
  for bit in info_bits:
    fb = ((reg >> (crc_length - 1)) ^ bit) & 1
    reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    reg &= mask

  crc_bits = np.array(
    [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 末尾 CRC 是否正确"""
  bits = np.asarray(bits, dtype=int)
  expected = crc_encode(bits[:-crc_length], crc_length)
  return np.array_equal(bits, expected)


class _Path:
    """SCL 单条路径（Lazy Copy）"""

    __slots__ = ("parent", "branch", "pm", "B_snapshot")

    def __init__(self, parent=None, branch=0, pm=0.0):
        self.parent = parent
        self.branch = branch  # 0 or 1 at split
        self.pm = pm
        self.B_snapshot = None


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_set = set(np.where(self.frozen_bits == 0)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        # 每条路径维护完整 SCD 状态
        paths = []
        state = _SCDState(N, self.frozen_set)
        state.set_channel_llr(llr_ch.copy())
        paths.append({"state": state, "pm": 0.0, "u": np.zeros(N, dtype=int)})

        for i in range(N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                st = path["state"]
                st._update_llrs(l)
                llr_val = st.L[l, self.n]
                if np.isnan(llr_val):
                    llr_val = 0.0

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    st.B[l, self.n] = 0
                    st._update_bits(l)
                    path["u"][l] = 0
                    new_paths.append(
                        {"state": st, "pm": path["pm"] + penalty, "u": path["u"].copy()}
                    )
                else:
                    for bit in (0, 1):
                        st_copy = self._copy_state(st)
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (
                            bit == 1 and llr_val < 0
                        ) else abs(llr_val)
                        st_copy.B[l, self.n] = bit
                        st_copy._update_bits(l)
                        u_new = path["u"].copy()
                        u_new[l] = bit
                        new_paths.append(
                            {
                                "state": st_copy,
                                "pm": path["pm"] + penalty,
                                "u": u_new,
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        # 选择最优路径
        if self.crc_length > 0:
            info_indices = sorted(self.info_set)
            valid = []
            for p in paths:
                info_bits = p["u"][info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]

    def _copy_state(self, st):
        new_st = _SCDState(self.N, self.frozen_set)
        new_st.L = st.L.copy()
        new_st.B = st.B.copy()
        return new_st
