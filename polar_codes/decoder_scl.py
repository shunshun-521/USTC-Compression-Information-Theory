"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr, _frozen_to_set,
)
from encoder import bit_reversal_permutation


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（MSB-first，poly 含隐式最高位）"""
    bits = np.asarray(bits, dtype=int)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    r=8: CRC-8 (0x07); r=16: CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    expected = _crc_remainder(info, poly, crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = _frozen_to_set(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat (长度 N), pm (最优路径度量)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llrs_init = llr_ch[rev].copy()

        n = self.n
        N = self.N
        L_size = self.list_size

        # 路径状态
        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.full((N, n + 1), np.nan),
            'pm': 0.0,
        }]
        paths[0]['L'][:, 0] = llrs_init

        def update_llrs_path(path, l):
            L = path['L']
            B = path['B']
            for s in range(n - _active_llr_level(l, n), n):
                block_size = int(2 ** (s + 1))
                branch_size = block_size // 2
                for j in range(l, N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = _lower_llr(
                            L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                        )

        def update_bits_path(path, l):
            if l < N / 2:
                return
            L = path['L']
            B = path['B']
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = int(2 ** s)
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            # 所有路径更新 LLR
            for path in paths:
                update_llrs_path(path, l)
                llr_val = path['L'][l, n]

                if l in self.frozen_set:
                    # 冻结位
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    path['pm'] += penalty
                    path['B'][l, n] = 0
                    update_bits_path(path, l)
                else:
                    path['candidates'] = []
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_val >= 0) or (bit == 1 and llr_val < 0) else abs(llr_val)
                        path['candidates'].append((bit, path['pm'] + penalty))

            # 扩展路径
            new_paths = []
            for path in paths:
                if l in self.frozen_set:
                    new_paths.append(path)
                else:
                    for bit, pm in path['candidates']:
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': pm,
                        }
                        new_path['B'][l, n] = bit
                        update_bits_path(new_path, l)
                        new_paths.append(new_path)

            # 裁剪到 L_size
            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L_size]

        # 选择最优路径
        crc_valid = []
        for path in paths:
            u_hat = path['B'][:, n].astype(int)
            if self.crc_length > 0:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)
        if crc_valid:
            best = min(crc_valid, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['B'][:, n].astype(int), best['pm']
