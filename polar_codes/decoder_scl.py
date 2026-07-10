"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly, mask = CRC8_POLY, 0xFF
    elif crc_length == 16:
        poly, mask = CRC16_POLY, 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = _Path(self.L.shape[0], self.L.shape[1] - 1)
        new.L[:] = self.L
        new.B[:] = self.B
        new.pm = self.pm
        new.u_hat[:] = self.u_hat
        return new


class SCLDecoder:
    """SCL 译码器（含 CRC 辅助 CA-SCL）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits.astype(bool))[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def _update_llrs(self, paths, l):
        for path in paths:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 1 << (s + 1)
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

    def _update_bits(self, paths, l):
        if l < self.N // 2:
            return
        for path in paths:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        init = _Path(self.N, self.n)
        init.L[:, 0] = llr_ch[self.brp]
        paths = [init]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(paths, l)
            llr_dec = paths[0].L[l, self.n]

            if l in self.frozen_set:
                new_paths = []
                for path in paths:
                    np_ = path.copy()
                    np_.pm += _path_metric_penalty(path.L[l, self.n], 0)
                    np_.u_hat[l] = 0
                    np_.B[l, self.n] = 0
                    new_paths.append(np_)
                paths = new_paths
            else:
                candidates = []
                for path in paths:
                    llr = path.L[l, self.n]
                    for bit in (0, 1):
                        np_ = path.copy()
                        np_.pm += _path_metric_penalty(llr, bit)
                        np_.u_hat[l] = bit
                        np_.B[l, self.n] = bit
                        candidates.append(np_)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[: self.list_size]

            self._update_bits(paths, l)

        if self.crc_length > 0:
            crc_ok = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            if crc_ok:
                paths = crc_ok

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0, num_frames=50, seed=1):
    """L=1 的 SCL 应与 SC 一致。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(seed)
    rate = K / N
    sigma = 1.0 / np.sqrt(2 * rate * (10 ** (eb_n0_db / 10.0)))
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL L=1 verification passed.")
