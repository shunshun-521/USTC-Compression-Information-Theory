"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _as_frozen_set,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """单条 SCL 路径（Lazy Copy）。"""

    __slots__ = ("L", "B", "pm", "parent", "copy_L", "copy_B")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.parent = None
        self.copy_L = False
        self.copy_B = False

    def fork(self):
        child = Path.__new__(Path)
        child.L = self.L
        child.B = self.B
        child.pm = self.pm
        child.parent = self
        child.copy_L = False
        child.copy_B = False
        return child

    def ensure_L_writable(self):
        if self.parent is not None and not self.copy_L:
            self.L = self.L.copy()
            self.copy_L = True

    def ensure_B_writable(self):
        if self.parent is not None and not self.copy_B:
            self.B = self.B.copy()
            self.copy_B = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _as_frozen_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = sorted(set(range(N)) - self.frozen_set)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], path.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, bit):
        preferred = 0 if llr >= 0 else 1
        return 0.0 if bit == preferred else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.N, self.n, llr_ch)]

        for phi in [bit_reversed(i, self.n) for i in range(self.N)]:
            candidates = []

            for path in paths:
                path.ensure_L_writable()
                self._update_llrs(path, phi)
                llr = path.L[phi, self.n]

                if phi in self.frozen_set:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.ensure_B_writable()
                    path.B[phi, self.n] = 0
                    path.pm += penalty
                    self._update_bits(path, phi)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = path.fork()
                        child.pm += self._path_metric_penalty(llr, bit)
                        child.ensure_B_writable()
                        child.B[phi, self.n] = bit
                        self._update_bits(child, phi)
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best = paths[0]
        u_hat = best.B[:, self.n].astype(int)

        if self.crc_length > 0:
            info_bits = u_hat[self.info_indices]
            valid = [
                p for p in paths if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
                u_hat = best.B[:, self.n].astype(int)

        return u_hat, best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(15.0, K / N)
    mismatches = 0
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 不一致: {mismatches}"
    print("SCL L=1 与 SC 一致性校验通过")
