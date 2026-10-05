"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _leaf_start(depth, node, n):
    return node * (2 ** (n - 1 - depth))


def _pm_penalty(llr_val, bit):
    hard = 1 if llr_val < 0 else 0
    return abs(llr_val) if bit != hard else 0.0


def _channel_score(u, llr):
    """与 BPSK LLR 一致的软度量（越小越好）。"""
    from encoder import polar_encode

    x = polar_encode(u)
    s = 1.0 - 2.0 * x
    z = np.clip(-llr * s, -60.0, 60.0)
    return float(np.sum(np.log1p(np.exp(z))))


def scl_decode_recursive(llr, frozen_bits, list_size):
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N)) + 1
    frozen_set = set(np.where(frozen_bits)[0])
    paths = [(np.zeros(N, dtype=int), 0.0)]

    def prune(cands):
        return sorted(
            cands, key=lambda x: (x[1], _channel_score(x[0], llr))
        )[:list_size]

    def dfs(y, depth, node):
        nonlocal paths
        if depth == n - 1:
            expanded = []
            for nv, pm in paths:
                if node in frozen_set:
                    nv2 = nv.copy()
                    nv2[node] = 0
                    expanded.append((nv2, pm + _pm_penalty(y[0], 0)))
                else:
                    for bit in (0, 1):
                        nv2 = nv.copy()
                        nv2[node] = bit
                        expanded.append((nv2, pm + _pm_penalty(y[0], bit)))
            paths[:] = prune(expanded)
            return

        half = len(y) // 2
        L1, L2 = y[:half], y[half:]
        left_llr = f_operation(L1, L2)
        dfs(left_llr, depth + 1, 2 * node)

        merged = []
        for nv, pm in list(paths):
            ls = _leaf_start(depth + 1, 2 * node, n)
            arr1 = nv[ls : ls + half]
            right_llr = g_operation(L1, L2, arr1)
            paths[:] = [(nv, pm)]
            dfs(right_llr, depth + 1, 2 * node + 1)
            merged.extend(paths)
        paths[:] = prune(merged)

    dfs(llr, 0, 0)
    return min(paths, key=lambda x: (x[1], _channel_score(x[0], llr)))


def _fast_scl(llr, frozen_bits, list_size, info_indices):
    """大码长下的快速近似 SCL：SC + 单比特翻转候选 + 信道软度量。"""
    u_sc = sc_decode_recursive(llr, frozen_bits)
    candidates = {tuple(u_sc): _channel_score(u_sc, llr)}
    order = sorted(info_indices, key=lambda i: abs(llr[i]))
    for idx in order:
        if len(candidates) >= list_size:
            break
        for bit in (0, 1):
            if bit == u_sc[idx]:
                continue
            u = u_sc.copy()
            u[idx] = bit
            key = tuple(u)
            if key not in candidates:
                candidates[key] = _channel_score(u, llr)
    best_u = min(candidates.items(), key=lambda x: x[1])[0]
    return np.array(best_u, dtype=int), float(candidates[tuple(best_u)])


class SCLDecoder:
    """SCL 译码器（含 CRC 辅助 CA-SCL）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        if self.N > 128 and self.list_size > 1:
            u_hat, pm = _fast_scl(
                llr_ch, self.frozen_bits, self.list_size, self.info_indices
            )
        else:
            u_hat, pm = scl_decode_recursive(
                llr_ch, self.frozen_bits, self.list_size
            )

        if self.crc_length > 0:
            info = u_hat[self.info_indices]
            if not crc_check(info, self.crc_length):
                if self.N > 128:
                    u2, pm2 = _fast_scl(
                        llr_ch,
                        self.frozen_bits,
                        min(self.list_size * 4, 32),
                        self.info_indices,
                    )
                else:
                    u2, pm2 = scl_decode_recursive(
                        llr_ch, self.frozen_bits, min(self.list_size * 4, 32)
                    )
                if crc_check(u2[self.info_indices], self.crc_length):
                    return u2, pm2
        return u_hat, pm
