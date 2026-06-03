"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


# CRC-8: 0x07, CRC-16: 0x8005
_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC_POLY[crc_length]
    reg = 0
    for b in info_bits:
        reg ^= (int(b) << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat", "parent", "branch_u")

    def __init__(self, N, n, llr_ch, parent=None, branch_u=0):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch if parent is None else parent.L[:, 0]
        self.B = parent.B.copy() if parent is not None else self.B
        self.L = parent.L.copy() if parent is not None else self.L
        self.pm = parent.pm if parent is not None else 0.0
        self.u_hat = parent.u_hat.copy() if parent is not None else np.zeros(N, dtype=int)
        self.parent = parent
        self.branch_u = branch_u


class SCLDecoder:
    """SCL 译码器（路径复制 + 列表裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        """与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [_Path(N, n, llr_ch)]
        paths[0].pm = 0.0

        for i in range(N):
            l = bit_reversed_index(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                cur_llr = path.L[l, n]

                if l in self.frozen_set:
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    path.pm += self._pm_penalty(cur_llr, 0)
                    _update_bits(path.B, l, n, N)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        child = _Path(N, n, llr_ch, parent=path, branch_u=u)
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        child.pm = path.pm + self._pm_penalty(cur_llr, u)
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(p)
            else:
                crc_ok.append(p)

        if crc_ok:
            best = min(crc_ok, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.astype(int), best.pm


def verify_scl_equals_sc(N=32, K=16, eb_n0_db=8.0, num_frames=20):
    """L=1 的 SCL 应与 SC 一致（无 CRC）。"""
    from construction import ga_construction
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            raise AssertionError("SCL L=1 != SC")
