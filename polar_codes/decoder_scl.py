"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _calc_llr, _llr_check_node, sc_decode


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = np.array([1, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    elif crc_length == 16:
        poly = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")

    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    while np.any(msg[: len(info_bits)]):
        start = int(np.argmax(msg[: len(info_bits)] != 0))
        msg[start : start + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[len(info_bits) :]])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    return np.array_equal(crc_encode(info, crc_length)[-crc_length:], bits[-crc_length:])


class _Path:
    __slots__ = ("pm", "u_hat", "active")

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 u_hat 数组拷贝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _path_llr(self, llr_ch, u_prefix):
        """计算当前比特的 LLR（复用 SC 递归）"""
        i = len(u_prefix)
        return _calc_llr(i, self.n, llr_ch, u_prefix)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N)]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                if not path.active:
                    continue
                llr_phi = self._path_llr(llr_ch, path.u_hat[:phi])

                if self.frozen_bits[phi]:
                    penalty = 0.0 if llr_phi >= 0 else abs(llr_phi)
                    path.pm += penalty
                    path.u_hat[phi] = 0
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N)
                        p.u_hat = path.u_hat.copy()
                        p.pm = path.pm
                        p.u_hat[phi] = bit
                        expected = 0 if llr_phi >= 0 else 1
                        if bit != expected:
                            p.pm += abs(llr_phi)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, seed=0):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.zeros(N, dtype=int)
    frozen[np.setdiff1d(np.arange(N), info_idx)] = 1

    rng = np.random.default_rng(seed)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(8.0, K / N)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1, crc_length=0).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
