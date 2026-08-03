"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB first）。"""
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _b_check(layer, idx):
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _li(layer, idx, llrs, s):
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s),
            _li(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s),
            _li(layer + 1, idx, llrs, s),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


class SCLDecoder:
    """SCL 译码器（Lazy Copy 路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        L = self.list_size

        if L == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llrs = []
        bits = []
        for _ in range(L):
            llr_arr = -np.inf * np.ones((self.n + 1, N), dtype=np.float64)
            llr_arr[self.n, :] = llr_ch
            bit_arr = -np.ones((self.n + 1, N), dtype=int)
            llrs.append(llr_arr)
            bits.append(bit_arr)

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for bit_idx in range(N):
            dm = np.zeros(L, dtype=np.float64)

            for path in range(L):
                if pm[path] == np.inf and path > 0:
                    continue
                current_llr = _li(0, bit_idx, llrs[path], bits[path])
                llrs[path][0, bit_idx] = current_llr

                if self.frozen_bits[bit_idx]:
                    bits[path][0, bit_idx] = 0
                    pm[path] += abs(current_llr) if current_llr < 0 else 0.0
                else:
                    hard = 0 if current_llr >= 0 else 1
                    bits[path][0, bit_idx] = hard
                    dm[path] = abs(current_llr)

            if (not self.frozen_bits[bit_idx]) and L > 1:
                pm_dm = np.concatenate([pm, pm + dm])
                order = np.argsort(pm_dm)
                selected = order[:L]

                new_llrs = [None] * L
                new_bits = [None] * L
                new_pm = np.full(L, np.inf, dtype=np.float64)

                for new_i, global_i in enumerate(selected):
                    src = global_i % L
                    flip = global_i >= L
                    new_llrs[new_i] = llrs[src].copy()
                    new_bits[new_i] = bits[src].copy()
                    if flip:
                        new_bits[new_i][0, bit_idx] = 1 - new_bits[new_i][0, bit_idx]
                    new_pm[new_i] = pm_dm[global_i]

                llrs = new_llrs
                bits = new_bits
                pm = new_pm

        candidates = []
        for path in range(L):
            if pm[path] == np.inf:
                continue
            u_hat = bits[path][0, :].copy()
            u_hat[u_hat < 0] = 0
            candidates.append((pm[path], u_hat))

        if not candidates:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        if self.crc_length > 0:
            crc_paths = [
                (p, u) for p, u in candidates
                if crc_check(u[self.info_positions], self.crc_length)
            ]
            if crc_paths:
                candidates = crc_paths

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u.astype(int), float(best_pm)
