"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _permute_channel_llr,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """单条 SCL 译码路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        new_path = Path(self.L.shape[0], int(math.log2(self.L.shape[0])))
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, paths, l):
        for path in paths:
            start_layer = self.n - _active_llr_level(l, self.n)
            for s in range(start_layer, self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        path.L[j, s + 1] = f_operation(
                            path.L[j, s], path.L[j + branch_size, s]
                        )
                    else:
                        top_bit = path.B[j - branch_size, s + 1]
                        path.L[j, s + 1] = g_operation(
                            path.L[j - branch_size, s], path.L[j, s], top_bit
                        )

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        end_layer = self.n - _active_bit_level(l, self.n)
        for path in paths:
            for s in range(self.n, end_layer, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = (
                            path.B[j, s] ^ path.B[j - branch_size, s]
                        ) & 1
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = _permute_channel_llr(llr_ch)
        paths = [Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            self._update_llrs(paths, l)
            llr = paths[0].L[l, self.n]

            if l in self.frozen_set:
                for path in paths:
                    llr_val = path.L[l, self.n]
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    path.pm += self._path_metric_penalty(llr_val, 0)
                self._update_bits(paths, l)
                continue

            candidates = []
            for path in paths:
                llr_val = path.L[l, self.n]
                for bit in (0, 1):
                    new_path = path.copy()
                    new_path.u_hat[l] = bit
                    new_path.B[l, self.n] = bit
                    new_path.pm += self._path_metric_penalty(llr_val, bit)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]
            self._update_bits(paths, l)

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.u_hat.astype(int), best.pm


def verify_scl_equals_sc():
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed")
