"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于 Permuted SCD
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)


def _crc_division(info_bits, poly, crc_length):
    bits = np.concatenate([np.asarray(info_bits, dtype=int), np.zeros(crc_length, dtype=int)])
    for i in range(len(info_bits)):
        if bits[i]:
            for j, p in enumerate(poly):
                if p:
                    bits[i + j] ^= 1
    return bits[-crc_length:]


_CRC_POLYS = {
    8: np.array([1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=int),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int),
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    if crc_length not in _CRC_POLYS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    info_bits = np.asarray(info_bits, dtype=int)
    crc_bits = _crc_division(info_bits, _CRC_POLYS[crc_length], crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    if crc_length not in _CRC_POLYS:
        return False
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        N, n = self.L.shape[0], self.L.shape[1] - 1
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.phase_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.phase_order:
            candidates = []
            for path in paths:
                self._update_llrs(path.L, path.B, l)
                cur_llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(cur_llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    self._update_bits(path.B, l)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._pm_penalty(cur_llr, bit)
                        new_path.u_hat[l] = bit
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path.B, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            for p in paths:
                if crc_check(p.u_hat[self.info_indices], self.crc_length):
                    if best_crc is None or p.pm < best_crc.pm:
                        best_crc = p

        chosen = best_crc if best_crc is not None else best_any
        return chosen.u_hat.copy(), chosen.pm
