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
    _get_sc_cache,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    msb = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class Path:
    """单条译码路径，支持 lazy copy。"""

    __slots__ = ("pm", "u_hat", "P", "C", "P_ref", "C_ref")

    def __init__(self, n, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.P = [np.zeros((N, n + 1), dtype=np.float64) for _ in range(1)]
        self.C = [np.zeros((N, n + 1), dtype=int) for _ in range(1)]
        self.P_ref = [0]
        self.C_ref = [0]

    def clone(self):
        new = Path(0, len(self.u_hat))
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        new.P = [self.get_P().copy()]
        new.C = [self.get_C().copy()]
        new.P_ref = [0]
        new.C_ref = [0]
        return new

    def get_P(self):
        return self.P[self.P_ref[0]]

    def get_C(self):
        return self.C[self.C_ref[0]]

    def ensure_copy(self):
        if self.P_ref[0] != 0:
            self.P[0] = self.P[self.P_ref[0]].copy()
            self.P_ref[0] = 0
        if self.C_ref[0] != 0:
            self.C[0] = self.C[self.C_ref[0]].copy()
            self.C_ref[0] = 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, self.llr_layer_vec, self.bit_layer_vec, self.decode_order = _get_sc_cache(N)

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llr(self, path, step, l):
        P = path.get_P()
        C = path.get_C()
        for s in self.llr_layer_vec[step]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    P[j, s + 1] = f_operation(P[j, s], P[j + branch_size, s])
                else:
                    P[j, s + 1] = g_operation(
                        P[j - branch_size, s], P[j, s], C[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, step, l):
        P = path.get_P()
        C = path.get_C()
        for s in self.bit_layer_vec[step]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    C[j - branch_size, s - 1] = C[j, s] ^ C[j - branch_size, s]
                    C[j, s - 1] = C[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_pc = llr_ch[br]

        paths = [Path(self.n, self.N)]
        paths[0].get_P()[:, 0] = llr_pc

        for step, l in enumerate(self.decode_order):
            new_paths = []
            for path in paths:
                path.ensure_copy()
                self._update_llr(path, step, l)
                llr = path.get_P()[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._path_metric_penalty(llr, 0)
                    path.u_hat[l] = 0
                    path.get_C()[l, self.n] = 0
                    self._update_bits(path, step, l)
                    new_paths.append(path)
                else:
                    p1 = path.clone()
                    for p, u in ((path, 0), (p1, 1)):
                        p.pm += self._path_metric_penalty(llr, u)
                        p.u_hat[l] = u
                        p.get_C()[l, self.n] = u
                        self._update_bits(p, step, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    mismatches = 0
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 SCL vs SC: {50 - mismatches}/50 一致")
    assert mismatches == 0
