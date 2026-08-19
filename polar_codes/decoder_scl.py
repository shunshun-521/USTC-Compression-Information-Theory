"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reverse_llr
from decoder_sc import (
    f_operation, g_operation, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
    else:
        reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_step(reg, b, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = bit_reverse_llr(llr_ch)
        N, n = self.N, self.n

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=int),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                L, B, pm, u_hat = path['L'], path['B'], path['pm'], path['u_hat']

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            top_bit = B[j - branch_size, s + 1]
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s], L[j, s], top_bit
                            )

                llr = L[l, n]

                if self.frozen_bits[l]:
                    B[l, n] = 0
                    u_hat[l] = 0
                    pm_new = pm + self._pm_penalty(llr, 0)
                    new_paths.append({
                        'L': L.copy(),
                        'B': B.copy(),
                        'pm': pm_new,
                        'u_hat': u_hat.copy(),
                    })
                else:
                    for u in (0, 1):
                        Lc, Bc, uhc = L.copy(), B.copy(), u_hat.copy()
                        Bc[l, n] = u
                        uhc[l] = u
                        new_paths.append({
                            'L': Lc,
                            'B': Bc,
                            'pm': pm + self._pm_penalty(llr, u),
                            'u_hat': uhc,
                        })

            for path in new_paths:
                l = _bit_reversed(phi, n)
                if l < N // 2:
                    continue
                B, u_hat = path['B'], path['u_hat']
                for s in range(n, n - _active_bit_level(l, n), -1):
                    block_size = 1 << s
                    branch_size = block_size // 2
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                            B[j, s - 1] = B[j, s]

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        best_crc = None
        best_all = min(paths, key=lambda p: p['pm'])
        if self.crc_length > 0:
            for p in paths:
                info_part = p['u_hat'][self.info_idx]
                if crc_check(info_part, self.crc_length):
                    if best_crc is None or p['pm'] < best_crc['pm']:
                        best_crc = p
        chosen = best_crc if best_crc is not None else best_all
        return chosen['u_hat'], chosen['pm']
