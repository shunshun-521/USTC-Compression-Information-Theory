"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    sc_decode, sc_decode_with_llrs, f_operation, g_operation, path_metric_update,
    _polar_decode_sc,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask)
        if feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY if crc_length == 16 else None
    if poly is None:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


def _llr_at_bit(llr_ch, frozen_bits, u_prefix, phi):
    """计算第 phi 个比特 LLR，已知 u_prefix[0:phi]。"""
    llr = np.asarray(llr_ch, dtype=np.float64).copy()
    frozen = np.asarray(frozen_bits, dtype=bool)

    def recurse(node_llr, frozen_node, offset):
        n = len(node_llr)
        if n == 1:
            idx = offset
            if idx == phi:
                return node_llr[0]
            return None

        half = n // 2
        left_llr = f_operation(node_llr[:half], node_llr[half:])

        if phi < offset + half:
            return recurse(left_llr, frozen_node[:half], offset)

        u_left_up = np.zeros(half, dtype=int)
        for i in range(half):
            idx = offset + i
            if idx < phi:
                u_left_up[i] = u_prefix[idx]
            else:
                _, u_up = _polar_decode_sc(left_llr, frozen_node[:half])
                u_left_up = u_up.copy()
                for j in range(half):
                    jdx = offset + j
                    if jdx < phi:
                        u_left_up[j] = u_prefix[jdx]
                break

        right_llr = g_operation(node_llr[:half], node_llr[half:], u_left_up)
        return recurse(right_llr, frozen_node[half:], offset + half)

    return recurse(llr, frozen, 0)


class _Path:
    __slots__ = ('pm', 'u_hat')

    def __init__(self, N):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat, pm = sc_decode(llr_ch, self.frozen_bits), 0.0
            return u_hat, pm

        paths = [_Path(self.N)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_val = _llr_at_bit(llr_ch, self.frozen_bits, path.u_hat, phi)
                if self.frozen_bits[phi]:
                    candidates.append((path_metric_update(path.pm, llr_val, 0), path, 0))
                else:
                    for u_cand in (0, 1):
                        candidates.append(
                            (path_metric_update(path.pm, llr_val, u_cand), path, u_cand)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:self.list_size]

            new_paths = []
            for new_pm, old, u_cand in candidates:
                p = _Path(self.N)
                p.pm = new_pm
                p.u_hat = old.u_hat.copy()
                p.u_hat[phi] = u_cand
                new_paths.append(p)
            paths = new_paths

        if self.crc_length > 0:
            crc_pass = [
                i for i, p in enumerate(paths)
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best_idx = min(crc_pass or range(len(paths)), key=lambda i: paths[i].pm)
        else:
            best_idx = min(range(len(paths)), key=lambda i: paths[i].pm)

        return paths[best_idx].u_hat.copy(), paths[best_idx].pm
