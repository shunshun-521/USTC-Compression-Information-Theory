"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed,
    sc_decode,
)


# CRC-8: x^8 + x^2 + x + 1 (0x07)
_CRC8_POLY = 0x07
# CRC-16: x^16 + x^15 + x^2 + 1 (0x8005)
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


def _pm_update(pm, llr, u):
    """路径度量更新：与 LLR 符号一致不惩罚，否则加 |LLR|"""
    penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
    return pm + penalty


class _Path:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（分层存储，路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [_Path(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if self.frozen_bits[l]:
                    u = 0
                    pm = _pm_update(path.pm, llr, u)
                    candidates.append((pm, pidx, u, False))
                else:
                    for u in (0, 1):
                        pm = _pm_update(path.pm, llr, u)
                        candidates.append((pm, pidx, u, True))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, pidx, u, _ in candidates[: self.list_size]:
                new_path = self._copy_path(paths[pidx])
                new_path.pm = pm
                new_path.B[l, self.n] = u
                new_path.u_hat[l] = u
                self._update_bits(new_path, l)
                new_paths.append(new_path)
            paths = new_paths

        best = self._select_best_path(paths)
        return best.u_hat.astype(int), best.pm

    def _copy_path(self, path):
        new_path = _Path(self.N, self.n, path.L[:, 0])
        new_path.L[:] = path.L
        new_path.B[:] = path.B
        new_path.u_hat[:] = path.u_hat
        new_path.pm = path.pm
        return new_path

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            passed = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    passed.append(p)
            if passed:
                return min(passed, key=lambda p: p.pm)
        return min(paths, key=lambda p: p.pm)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)

    u_sc, _ = sc_decode(llr, frozen_bits), 0.0
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("L=1 matches SC:", np.array_equal(u_sc, u_scl))

    u_scl4, _ = SCLDecoder(N, frozen_bits, list_size=4).decode(llr)
    print("SCL L=4 correct:", np.array_equal(u_scl4[info_idx], u[info_idx]))
