"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    LLR_CLIP,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_channel_llrs,
)


def _clip_llr(val):
    return float(np.clip(val, -LLR_CLIP, LLR_CLIP))


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _PathState:
    """单条译码路径状态。"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new = _PathState.__new__(_PathState)
        new.L = self.L.copy()
        new.B = self.B.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（Lazy Copy 通过路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _clip_llr(
                        f_operation(path.L[j, s], path.L[j + branch_size, s])
                    )
                else:
                    path.L[j, s + 1] = _clip_llr(
                        g_operation(
                            path.L[j - branch_size, s],
                            path.L[j, s],
                            path.B[j - branch_size, s + 1],
                        )
                    )

    def _update_bits(self, path, l, bit):
        path.u_hat[l] = bit
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] + path.B[j - branch_size, s]
                    ) % 2
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = _prepare_channel_llrs(llr_ch)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits, _llr_prepared=True), 0.0

        paths = [_PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.copy()
                    if llr0 < 0:
                        new_path.pm += abs(llr0)
                    self._update_bits(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    hard = 0 if llr0 >= 0 else 1
                    for bit in (0, 1):
                        new_path = path.copy()
                        if bit != hard:
                            new_path.pm += abs(llr0)
                        self._update_bits(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_ok = []
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(path)
            best = min(crc_ok if crc_ok else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    mismatches = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/20")
