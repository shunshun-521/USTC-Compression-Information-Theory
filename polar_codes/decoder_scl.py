"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, _bit_reversed, _active_llr_level,
    _active_bit_level, sc_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB-first）"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _update_llrs_path(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits_path(B, l, n):
    N = B.shape[0]
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        rev = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[rev]
        N, n, L = self.N, self.n, self.list_size

        paths = [{
            'pm': 0.0,
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.float64),
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs_path(path['L'], path['B'], l, n)
                llr = path['L'][l, n]

                if l in self.frozen_set:
                    pm = self._pm_update(path['pm'], llr, 0)
                    new_p = {
                        'pm': pm,
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_p['u_hat'][l] = 0
                    new_p['B'][l, n] = 0
                    _update_bits_path(new_p['B'], l, n)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path['pm'], llr, u)
                        new_p = {
                            'pm': pm,
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_p['u_hat'][l] = u
                        new_p['B'][l, n] = u
                        _update_bits_path(new_p['B'], l, n)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]

        if self.crc_length > 0:
            valid = [p for p in paths
                     if crc_check(p['u_hat'][self.info_indices], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else paths[0]
        else:
            best = paths[0]

        return best['u_hat'], best['pm']
