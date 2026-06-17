"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _SCDCore,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = np.zeros(crc_length + 1, dtype=np.int8)
    for coeff in range(crc_length + 1):
        if (poly >> (crc_length - coeff)) & 1:
            reg[coeff] = 1

    state = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        feedback = int(bit) ^ state[0]
        state[:-1] = state[1:]
        state[-1] = 0
        if feedback:
            state ^= reg[1:]
    return state


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    remainder = _crc_remainder(bits, crc_length)
    return not np.any(remainder)


class _Path:
  __slots__ = ("core", "pm", "u_hat")

  def __init__(self, N, frozen_bits):
      self.core = _SCDCore(N, frozen_bits)
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 SC 核心状态）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, bit):
        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.frozen_bits) for _ in range(1)]
        paths[0].core.L[:, 0] = llr_ch.copy()

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                path.core._update_llrs(l)
                llr = path.core.L[l, self.n]
                if np.isnan(llr):
                    llr = 0.0

                if self.frozen_bits[phi]:
                    bit = 0
                    path.pm = self._pm_update(path.pm, llr, bit)
                    path.u_hat[phi] = bit
                    path.core.B[l, self.n] = bit
                    path.core._update_bits(l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = self._clone_path(path)
                        child.pm = self._pm_update(path.pm, llr, bit)
                        child.u_hat[phi] = bit
                        child.core.B[l, self.n] = bit
                        child.core._update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm

    def _clone_path(self, path):
        child = _Path(self.N, self.frozen_bits)
        child.pm = path.pm
        child.u_hat = path.u_hat.copy()
        child.core.L = path.core.L.copy()
        child.core.B = path.core.B.copy()
        return child
