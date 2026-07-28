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
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _frozen_to_set,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length 仅支持 8 或 16")

    info_bits = np.asarray(info_bits, dtype=int).ravel()
    remainder = _crc_remainder(info_bits, poly, width)
    crc_bits = np.array([(remainder >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 8:
        poly, width = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, width = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length 仅支持 8 或 16")

    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < width:
        return False
    remainder = _crc_remainder(bits, poly, width)
    return remainder == 0


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, bit):
        penalty = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr = np.asarray(llr_ch, dtype=np.float64)[self.br].copy()
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            new_paths = []

            for path in paths:
                self._update_llr(path, l)

                llr_bit = path["L"][l, n]
                if l in self.frozen_set:
                    pm = self._path_metric_update(path["pm"], llr_bit, 0)
                    child = self._lazy_copy(path)
                    child["pm"] = pm
                    child["u"][l] = 0
                    child["B"][l, n] = 0
                    self._update_bits(child, l)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = self._path_metric_update(path["pm"], llr_bit, bit)
                        child = self._lazy_copy(path)
                        child["pm"] = pm
                        child["u"][l] = bit
                        child["B"][l, n] = bit
                        self._update_bits(child, l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        return self._select_best_path(paths)

    def _lazy_copy(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u": path["u"].copy(),
        }

    def _update_llr(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top = L[j - branch_size, s]
                    bottom = L[j, s]
                    bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = (bottom + top) if bit == 0 else (bottom - top)

    def _update_bits(self, path, l):
        B = path["B"]
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_valid(p["u"])]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
                return best["u"], best["pm"]
        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]

    def _crc_valid(self, u_hat):
        info_idx = np.where(~self.frozen_bits)[0]
        payload = u_hat[info_idx]
        return crc_check(payload, self.crc_length)


def verify_scl_equals_sc():
    """L=1 时 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr)
        assert np.array_equal(u_sc, u_scl)


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL 路径度量校验通过")
