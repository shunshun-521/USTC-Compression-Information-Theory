"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    """单条 SCL 路径（Lazy Copy）"""

    __slots__ = ("L", "B", "pm", "parent", "copy_L", "copy_B")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.parent = None
        self.copy_L = True
        self.copy_B = True

    def ensure_L(self):
        if self.copy_L and self.parent is not None:
            self.L = self.parent.L.copy()
            self.copy_L = False

    def ensure_B(self):
        if self.copy_B and self.parent is not None:
            self.B = self.parent.B.copy()
            self.copy_B = False


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _branch_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = llr_ch.astype(np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = self.br_order[phi]
            candidates = []

            for path in paths:
                path.ensure_L()
                path.ensure_B()
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    pen = self._branch_penalty(llr, 0)
                    new_path = _Path(self.N, self.n, llr_ch)
                    new_path.L = path.L.copy()
                    new_path.B = path.B.copy()
                    new_path.pm = path.pm + pen
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pen = self._branch_penalty(llr, bit)
                        new_path = _Path(self.N, self.n, llr_ch)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm + pen
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.B[:, self.n], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]

        return best.B[:, self.n].astype(int), best.pm
