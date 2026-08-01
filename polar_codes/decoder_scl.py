"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_encode
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = 0x07 if crc_length == 8 else 0x8005

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int32
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int32)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class _Path:
  __slots__ = ("L", "B", "pm", "u_hat", "active")

  def __init__(self, N, n, llr_ch):
    self.L = np.zeros((N, n + 1), dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=np.int32)
    self.L[:, 0] = llr_ch
    self.pm = 0.0
    self.u_hat = np.zeros(N, dtype=np.int32)
    self.active = True


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.decode_order = [int(self.br[i]) for i in range(N)]

    def _update_llrs(self, path, l):
        start_layer = self.n - _active_llr_level(l, self.n)
        for s in range(start_layer, self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l, u_val):
        path.B[l, self.n] = u_val
        path.u_hat[l] = u_val
        if l < self.N // 2:
            return
        start_bit = self.n - _active_bit_level(l, self.n) + 1
        for s in range(self.n, start_bit - 1, -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]

        paths = [_Path(self.N, self.n, llr_ch.copy())]

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if self.frozen_bits[l]:
                    penalty = self._pm_penalty(llr_val, 0)
                    new_path = self._copy_path(path)
                    new_path.pm += penalty
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr_val, u_val)
                        self._update_bits(new_path, l, u_val)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            if not paths:
                break

        best = self._select_best_path(paths)
        return best.u_hat.copy(), best.pm

    def _copy_path(self, path):
        """Lazy copy：仅复制必要状态（浅拷贝数组引用，分裂时深拷贝）。"""
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        return new_path

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[~self.frozen_bits]
                if len(info_bits) >= self.crc_length and crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)
