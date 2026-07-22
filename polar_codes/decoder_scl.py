"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _b_check, _compute_llr, _s_updater


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    elif crc_length == 16:
        poly = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    else:
        raise ValueError('crc_length must be 8 or 16')
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[-crc_length:]])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = np.array([1, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    else:
        poly = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
    msg = bits.copy()
    for i in range(len(bits) - crc_length + 1):
        if msg[i] == 1:
            msg[i:i + len(poly)] ^= poly
    return np.all(msg[-crc_length + 1:] == 0)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_mask = ~self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        N, n = self.N, self.n

        llrs = [np.full((n + 1, N), -np.inf, dtype=np.float64) for _ in range(L)]
        bits = [np.full((n + 1, N), -1, dtype=np.int8) for _ in range(L)]
        for l in range(L):
            llrs[l][n, :] = llr_ch

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for i in range(N):
            is_frozen = self.frozen_bits[i]
            metrics = np.zeros(L, dtype=np.float64)

            for l in range(L):
                llrs[l][0, i] = _compute_llr(0, i, llrs[l], bits[l])
                if is_frozen:
                    bits[l][0, i] = 0
                    pm[l] += 0.0 if llrs[l][0, i] >= 0 else abs(llrs[l][0, i])
                else:
                    bits[l][0, i] = 0 if llrs[l][0, i] >= 0 else 1
                    metrics[l] = abs(llrs[l][0, i])

            if not is_frozen and L > 1:
                pm_dm = np.concatenate([pm, pm + metrics])
                order = np.argsort(pm_dm)
                survivors = order[:L]
                new_llrs = [None] * L
                new_bits = [None] * L
                new_pm = np.zeros(L, dtype=np.float64)
                src_idx = 0
                for surv in survivors:
                    if surv < L:
                        new_llrs[src_idx] = llrs[surv]
                        new_bits[src_idx] = bits[surv]
                        new_pm[src_idx] = pm[surv]
                    else:
                        base = surv - L
                        new_llrs[src_idx] = llrs[base]
                        new_bits[src_idx] = bits[base].copy()
                        new_bits[src_idx][0, i] = 1 - bits[base][0, i]
                        new_pm[src_idx] = pm_dm[surv]
                    src_idx += 1
                llrs = new_llrs
                bits = new_bits
                pm = new_pm

        u_hat = bits[np.argmin(pm)][0, :].astype(int)

        if self.crc_length > 0:
            info_idx = np.where(self.info_mask)[0]
            valid = []
            for l in range(L):
                cand = bits[l][0, :].astype(int)
                payload = cand[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append((pm[l], cand))
            if valid:
                valid.sort(key=lambda x: x[0])
                u_hat = valid[0][1]

        return u_hat, float(np.min(pm))
