"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int_,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int_)
    if len(bits) < crc_length:
        return False
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(recomputed[-crc_length:], bits[-crc_length:])


class _Path:
  __slots__ = ('L', 'B', 'pm', 'u_hat')

  def __init__(self, N, n, llr_ch):
      self.L = np.zeros((N, n + 1), dtype=np.float64)
      self.B = np.zeros((N, n + 1), dtype=np.int_)
      self.L[:, 0] = llr_ch
      self.pm = 0.0
      self.u_hat = np.zeros(N, dtype=np.int_)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _llr_penalty(self, llr, bit):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 最优路径估计
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        inv_br = np.argsort(bit_reversal_permutation(self.N))
        llr_ch = llr_ch[inv_br]

        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    bit = 0
                    new_pm = path.pm + self._llr_penalty(llr, bit)
                    path.pm = new_pm
                    path.B[l, self.n] = bit
                    path.u_hat[l] = bit
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = _Path(self.N, self.n, llr_ch)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path.pm + self._llr_penalty(llr, bit)
                        child.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        self._update_bits(child, l)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p.u_hat)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _crc_valid(self, u_hat):
        """CRC 仅校验信息位（不含冻结位）。"""
        info_bits = u_hat[~self.frozen_bits]
        if len(info_bits) < self.crc_length:
            return False
        return crc_check(info_bits, self.crc_length)
