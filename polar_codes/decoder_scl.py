"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
import copy

from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    active_llr_level,
    active_bit_level,
    bit_reversed,
    _prepare_llr,
    _update_llrs,
    _update_bits,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_encode_poly(data_bits, crc_length):
    """GF(2) 多项式除法 CRC"""
    if crc_length == 8:
        poly = (1 << 8) | 0x07
    else:
        poly = (1 << 16) | 0x8005

    msg = 0
    for bit in data_bits:
        msg = (msg << 1) | int(bit)
    msg <<= crc_length

    deg = len(data_bits) + crc_length - 1
    for i in range(deg, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= poly << (i - crc_length)

    rem = msg & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(rem >> j) & 1 for j in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([data_bits, crc_bits]).astype(int)


def _crc_check_poly(bits, crc_length):
    if crc_length == 8:
        poly = (1 << 8) | 0x07
    else:
        poly = (1 << 16) | 0x8005

    msg = 0
    for bit in bits:
        msg = (msg << 1) | int(bit)

    deg = len(bits) - 1
    for i in range(deg, crc_length - 1, -1):
        if (msg >> i) & 1:
            msg ^= poly << (i - crc_length)

    return (msg & ((1 << crc_length) - 1)) == 0


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    return _crc_encode_poly(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if len(bits) < crc_length:
        return False
    return _crc_check_poly(bits, crc_length)


class Path:
  """SCL 单条路径（Lazy Copy 通过共享数组实现）"""

  __slots__ = ("L", "B", "pm", "u_hat")

  def __init__(self, N, n, llr):
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=int)
      self.L[:, 0] = llr
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=int)

  def copy(self):
      p = Path.__new__(Path)
      p.L = self.L.copy()
      p.B = self.B.copy()
      p.pm = self.pm
      p.u_hat = self.u_hat.copy()
      return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_update(self, pm, llr, bit):
        """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚"""
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n
        paths = [Path(N, n, llr)]

        for phase in range(N):
            l = bit_reversed(phase, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                cur_llr = path.L[l, n]

                if l in self.frozen_set:
                    p = path.copy()
                    p.B[l, n] = 0
                    p.u_hat[l] = 0
                    p.pm = self._pm_update(p.pm, cur_llr, 0)
                    _update_bits(p.B, l, n, N)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.B[l, n] = bit
                        p.u_hat[l] = bit
                        p.pm = self._pm_update(p.pm, cur_llr, bit)
                        _update_bits(p.B, l, n, N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[~self.frozen_bits]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm
