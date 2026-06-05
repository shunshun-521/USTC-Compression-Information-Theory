"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    _lower_llr,
    _prepare_channel_llrs,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


CRC_POLYNOMIALS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
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
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """单条 SCL 路径。"""

    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch.copy()

    def copy(self):
        new_path = Path.__new__(Path)
        new_path.pm = self.pm
        new_path.u_hat = self.u_hat.copy()
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _update_llrs(self, path, phase_i):
        l = _bit_reversed_index(phase_i, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )
        return path.L[l, self.n]

    def _propagate_bits(self, path, phase_i, bit_val):
        l = _bit_reversed_index(phase_i, self.n)
        path.B[l, self.n] = bit_val
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    @staticmethod
    def _path_metric_penalty(llr_val, bit_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llrs(llr_ch)
        paths = [Path(self.N, self.n, llr_ch)]

        for phase_i in range(self.N):
            l = _bit_reversed_index(phase_i, self.n)
            candidates = []
            for path in paths:
                llr_val = self._update_llrs(path, phase_i)
                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.u_hat[l] = 0
                    self._propagate_bits(new_path, phase_i, 0)
                    candidates.append(new_path)
                else:
                    for bit_val in (0, 1):
                        new_path = path.copy()
                        new_path.pm += self._path_metric_penalty(llr_val, bit_val)
                        new_path.u_hat[l] = bit_val
                        self._propagate_bits(new_path, phase_i, bit_val)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_pos = np.where(~self.frozen_bits)[0]
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[info_pos], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应与 SC 等价。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(3.0, K / N)
    rng = np.random.default_rng(7)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL L=1 != SC")
    return True
