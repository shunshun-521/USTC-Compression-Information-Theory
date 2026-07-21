"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    channel_llr,
    upper_llr,
    lower_llr,
    bit_reversed_index,
    active_llr_level,
    active_bit_level,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(extended, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC，返回 True/False"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    btm_llr = path.L[j, s]
                    top_llr = path.L[j - branch_size, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _clone_path(self, path):
        new_p = _Path(self.N, self.n)
        new_p.pm = path.pm
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        return new_p

    def _decode_paths(self, llr_ch, info_indices=None):
        llr_scd = channel_llr(llr_ch)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_scd

        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            new_paths = []
            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]
                branches = [0] if l in self.frozen_set else [0, 1]
                for u_bit in branches:
                    np_path = self._clone_path(path)
                    np_path.pm += self._path_metric_penalty(llr_val, u_bit)
                    np_path.B[l, self.n] = u_bit
                    self._update_bits(np_path, l)
                    new_paths.append(np_path)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0 and info_indices is not None:
            valid = [
                p for p in paths
                if crc_check(p.B[:, self.n][info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p.pm)
                return best.B[:, self.n].astype(int), best.pm

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm

    def decode(self, llr_ch):
        """SCL 译码，返回 (u_hat, pm)"""
        return self._decode_paths(llr_ch)

    def decode_with_info_indices(self, llr_ch, info_indices):
        """CA-SCL 译码"""
        return self._decode_paths(llr_ch, info_indices)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    mismatches = 0
    for _ in range(50):
        payload = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + np.random.normal(0, 0.01, N), 0.01
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"L=1 vs SC mismatches: {mismatches}/50")
