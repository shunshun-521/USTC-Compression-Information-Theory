"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _sc_decode_core, _bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr, f_operation, g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
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
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    expected = _crc_remainder(info, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received |= bits[-crc_length + i] << (crc_length - 1 - i)
    return expected == received


def _update_llrs(L, B, l, n, use_minsum=False):
    upper = f_operation if use_minsum else _upper_llr
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = upper(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = int(B[j - branch_size, s + 1])
                if use_minsum:
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)
                else:
                    L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _pm_penalty(llr_val, u):
    u_hard = 0 if llr_val >= 0 else 1
    return 0.0 if u == u_hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = []
        init_L = np.full((N, n + 1), np.nan, dtype=np.float64)
        init_B = np.full((N, n + 1), np.nan, dtype=np.float64)
        init_L[:, 0] = llr_ch
        paths.append({'L': init_L, 'B': init_B, 'pm': 0.0})

        for phase in range(N):
            l = _bit_reversed(phase, n)
            new_paths = []

            for path in paths:
                L = path['L'].copy()
                B = path['B'].copy()
                _update_llrs(L, B, l, n, use_minsum=True)
                cur_llr = L[l, n]

                if l in self.frozen_set:
                    new_paths.append({
                        'L': L, 'B': B,
                        'pm': path['pm'] + _pm_penalty(cur_llr, 0),
                        'bit': 0,
                        'l': l,
                    })
                else:
                    for u in (0, 1):
                        new_paths.append({
                            'L': L.copy(), 'B': B.copy(),
                            'pm': path['pm'] + _pm_penalty(cur_llr, u),
                            'bit': u,
                            'l': l,
                        })

            # 应用比特判决并回传
            expanded = []
            for p in new_paths:
                L, B = p['L'], p['B']
                l_idx = p['l']
                B[l_idx, n] = p['bit']
                _update_bits(B, l_idx, n, N)
                expanded.append({
                    'L': L, 'B': B, 'pm': p['pm'],
                    'decoded_bit': (l_idx, p['bit']),
                })

            expanded.sort(key=lambda x: x['pm'])
            paths = expanded[:L_size]

        # 组装 u_hat
        best_paths = []
        for p in paths:
            u_hat = p['B'][:, n].astype(np.int64)
            best_paths.append((p['pm'], u_hat))

        if self.crc_length > 0:
            crc_ok = [
                (pm, u) for pm, u in best_paths
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if crc_ok:
                pm, u_hat = min(crc_ok, key=lambda x: x[0])
            else:
                pm, u_hat = min(best_paths, key=lambda x: x[0])
        else:
            pm, u_hat = min(best_paths, key=lambda x: x[0])

        return u_hat, pm
