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
    prepare_llr_for_decode,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


# CRC 多项式
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB 优先）。"""
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
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _path_metric_update(self, pm, llr, u):
        """路径度量更新：与 LLR 符号不一致时加 |LLR|。"""
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if u == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm（最优路径度量）
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr0 = prepare_llr_for_decode(llr_ch, self.N)
        N, n = self.N, self.n
        Lsz = self.list_size

        # 路径状态：lazy copy 用父路径索引
        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1)),
            'B': np.zeros((N, n + 1), dtype=int),
            'parent': None,
            'u': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr0

        for i in range(N):
            l = _bit_reversed_index(i, n)
            new_paths = []

            for pidx, path in enumerate(paths):
                L = path['L']
                B = path['B']

                for s in range(n - _active_llr_level(l, n), n):
                    bs = 2 ** (s + 1)
                    br = bs // 2
                    for j in range(l, N, bs):
                        if j % bs < br:
                            L[j, s + 1] = f_operation(L[j, s], L[j + br, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j, s], L[j - br, s], B[j - br, s + 1]
                            )

                llr_bit = L[l, n]

                if self.frozen_bits[l]:
                    u_val = 0
                    pm = self._path_metric_update(path['pm'], llr_bit, u_val)
                    child = {
                        'pm': pm,
                        'L': L.copy(),
                        'B': B.copy(),
                        'parent': pidx,
                        'u': path['u'].copy(),
                    }
                    child['u'][l] = u_val
                    child['B'][l, n] = 0
                    if l >= N / 2:
                        for s in range(n, n - _active_bit_level(l, n), -1):
                            bs = 2 ** s
                            br = bs // 2
                            for j in range(l, -1, -bs):
                                if j % bs >= br:
                                    child['B'][j - br, s - 1] = (
                                        child['B'][j, s] ^ child['B'][j - br, s]
                                    )
                                    child['B'][j, s - 1] = child['B'][j, s]
                    new_paths.append(child)
                else:
                    for u_val in (0, 1):
                        pm = self._path_metric_update(path['pm'], llr_bit, u_val)
                        child = {
                            'pm': pm,
                            'L': L.copy(),
                            'B': B.copy(),
                            'parent': pidx,
                            'u': path['u'].copy(),
                        }
                        child['u'][l] = u_val
                        child['B'][l, n] = u_val
                        if l >= N / 2:
                            for s in range(n, n - _active_bit_level(l, n), -1):
                                bs = 2 ** s
                                br = bs // 2
                                for j in range(l, -1, -bs):
                                    if j % bs >= br:
                                        child['B'][j - br, s - 1] = (
                                            child['B'][j, s] ^ child['B'][j - br, s]
                                        )
                                        child['B'][j, s - 1] = child['B'][j, s]
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:Lsz]

        # 选择最优路径（CRC 辅助）
        info_positions = np.where(~self.frozen_bits)[0]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p['u'][info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        if crc_pass:
            best = min(crc_pass, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u'], best['pm']
