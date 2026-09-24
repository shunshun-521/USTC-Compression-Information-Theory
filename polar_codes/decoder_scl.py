"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _b_check, _s_updater, _li, _prepare_channel_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if feedback:
            reg ^= poly

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if feedback:
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        channel_llr = _prepare_channel_llr(llr_ch)
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = []
        s_list = []
        for _ in range(L):
            llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
            llrs[n, :] = channel_llr
            llrs_list.append(llrs)
            s_list.append(np.full((n + 1, N), -1, dtype=np.int8))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for phi in range(N):
            dm = np.zeros(L, dtype=np.float64)

            if self.frozen_bits[phi]:
                for l_idx in range(L):
                    llrs_list[l_idx][0, phi] = _li(0, phi, llrs_list[l_idx], s_list[l_idx], n)
                    s_list[l_idx][0, phi] = 0
                    if llrs_list[l_idx][0, phi] < 0:
                        pm[l_idx] += abs(llrs_list[l_idx][0, phi])
            else:
                for l_idx in range(L):
                    llrs_list[l_idx][0, phi] = _li(0, phi, llrs_list[l_idx], s_list[l_idx], n)
                    s_list[l_idx][0, phi] = 1 if llrs_list[l_idx][0, phi] < 0 else 0
                    dm[l_idx] = abs(llrs_list[l_idx][0, phi])

                if L > 1:
                    pm_dm = np.concatenate([pm, pm + dm])
                    idx_sort = np.argsort(pm_dm)[:L]

                    new_llrs = []
                    new_s = []
                    new_pm = np.zeros(L, dtype=np.float64)

                    for rank, idx in enumerate(idx_sort):
                        src = idx % L
                        bit = 0 if idx < L else 1
                        llrs_copy = llrs_list[src].copy()
                        s_copy = s_list[src].copy()
                        s_copy[0, phi] = bit
                        new_llrs.append(llrs_copy)
                        new_s.append(s_copy)
                        new_pm[rank] = pm_dm[idx]

                    llrs_list = new_llrs
                    s_list = new_s
                    pm = new_pm

        paths_u = [s_list[i][0, :].astype(int) for i in range(L)]

        if self.crc_length > 0:
            valid = []
            for i, u_hat in enumerate(paths_u):
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm[i], u_hat))
            if valid:
                best = min(valid, key=lambda x: x[0])
                return best[1], best[0]

        best_idx = int(np.argmin(pm))
        return paths_u[best_idx], pm[best_idx]
