"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, permute_llr_for_decode,
    _bit_reversed, _active_llr_level, _active_bit_level,
    _upper_llr, _lower_llr, precompute_sc_indices,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_generator(crc_length):
    if crc_length == 8:
        return (1 << 8) | _CRC8_POLY
    return (1 << 16) | _CRC16_POLY


def _crc_mod2_remainder(bits, crc_length):
    generator = _crc_generator(crc_length)
    msg = list(bits) + [0] * crc_length
    n = len(bits)
    for i in range(n):
        if msg[i]:
            for j in range(crc_length + 1):
                if generator & (1 << (crc_length - j)):
                    msg[i + j] ^= 1
    return np.array(msg[-crc_length:], dtype=np.int32)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    crc_bits = _crc_mod2_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int32)
    generator = _crc_generator(crc_length)
    msg = list(bits)
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i]:
            for j in range(crc_length + 1):
                if generator & (1 << (crc_length - j)):
                    msg[i + j] ^= 1
    return all(x == 0 for x in msg[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _path_metric_update(self, pm, llr_val, u):
        u_hard = 0 if llr_val >= 0 else 1
        penalty = 0.0 if u == u_hard else abs(llr_val)
        return pm + penalty

    def _update_llrs_path(self, L, B, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits_path(self, B, l):
        n, N = self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        N, n, L_size = self.N, self.n, self.list_size
        llr_perm = permute_llr_for_decode(llr_ch, N)

        paths = [{
            'L': np.full((N, n + 1), np.nan, dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=np.int32),
        }]
        paths[0]['L'][:, 0] = llr_perm

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                self._update_llrs_path(path['L'], path['B'], l)
                llr_val = path['L'][l, n]

                if self.frozen_bits[l]:
                    new_pm = self._path_metric_update(path['pm'], llr_val, 0)
                    candidates.append({
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'pm': new_pm,
                        'u_hat': path['u_hat'].copy(),
                        'u': 0,
                    })
                else:
                    for u in (0, 1):
                        new_pm = self._path_metric_update(path['pm'], llr_val, u)
                        candidates.append({
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'pm': new_pm,
                            'u_hat': path['u_hat'].copy(),
                            'u': u,
                        })

            candidates.sort(key=lambda x: x['pm'])
            candidates = candidates[:L_size]

            new_paths = []
            for cand in candidates:
                u = cand.pop('u')
                cand['u_hat'][l] = u
                cand['B'][l, n] = u
                self._update_bits_path(cand['B'], l)
                new_paths.append(cand)
            paths = new_paths

        if self.crc_length > 0:
            valid = [p for p in paths
                     if crc_check(p['u_hat'][self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda x: x['pm'])
        else:
            best = min(paths, key=lambda x: x['pm'])

        return best['u_hat'], best['pm']
