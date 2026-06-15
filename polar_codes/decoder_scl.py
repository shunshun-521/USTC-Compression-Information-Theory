"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _permute_channel_llr,
    _bit_reversed_int,
    _active_llr_level,
    _active_bit_level,
    precompute_sc_indices,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder_gf2(bits, crc_length, poly):
    """GF(2) 多项式长除求余数。"""
    poly_full = (1 << crc_length) | poly
    d = list(map(int, bits))
    for i in range(len(d) - crc_length):
        if d[i]:
            for j in range(crc_length + 1):
                if (poly_full >> (crc_length - j)) & 1:
                    d[i + j] ^= 1
    return np.array(d[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    augmented = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    crc_bits = _crc_remainder_gf2(augmented, crc_length, poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return np.all(_crc_remainder_gf2(bits, crc_length, poly) == 0)


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


def _path_update_llrs(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(
                    np.array([L[j, s]]), np.array([L[j + branch_size, s]])
                )[0]
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _path_update_bits(B, l, n):
    if l < 2 ** (n - 1):
        return
    N = B.shape[0]
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def decode(self, llr_ch):
        llr_ch = _permute_channel_llr(llr_ch)
        N, n, Lmax = self.N, self.n, self.list_size
        frozen = self.frozen_bits

        paths = [
            {
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "active": i == 0,
            }
            for i in range(Lmax)
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed_int(phi, n)
            candidates = []

            for path in paths:
                if not path["active"]:
                    continue
                L, B = path["L"], path["B"]
                _path_update_llrs(L, B, l, n)
                llr = L[l, n]

                if frozen[l]:
                    u = 0
                    candidates.append(
                        (path["pm"] + _pm_penalty(llr, u), path, u)
                    )
                else:
                    for u in (0, 1):
                        candidates.append(
                            (path["pm"] + _pm_penalty(llr, u), path, u)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:Lmax]

            new_paths = []
            for pm, parent, u in candidates:
                child = {
                    "pm": pm,
                    "u_hat": parent["u_hat"].copy(),
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "active": True,
                }
                child["u_hat"][l] = u
                child["B"][l, n] = u
                _path_update_bits(child["B"], l, n)
                new_paths.append(child)

            while len(new_paths) < Lmax:
                new_paths.append({"active": False})
            paths = new_paths

        active = [p for p in paths if p.get("active")]
        if not active:
            return np.zeros(N, dtype=int), 0.0

        if self.crc_length > 0:
            info_pos = np.where(~frozen)[0]
            passed = [p for p in active if crc_check(p["u_hat"][info_pos], self.crc_length)]
            pool = passed if passed else active
            best = min(pool, key=lambda p: p["pm"])
        else:
            best = min(active, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
