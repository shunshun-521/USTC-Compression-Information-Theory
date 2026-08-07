"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _li, INF
from encoder import channel_llr_to_decoder

CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)  # x^8+x^2+x+1
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = _crc_poly(crc_length)
    r = crc_length
    data = np.concatenate([np.asarray(info_bits, dtype=int), np.zeros(r, dtype=int)])
    for i in range(len(info_bits)):
        if data[i] == 1:
            data[i:i + len(poly)] ^= poly
    crc_bits = data[-r:]
    return np.concatenate([np.asarray(info_bits, dtype=int), crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC。"""
    poly = _crc_poly(crc_length)
    r = crc_length
    bits = np.asarray(bits, dtype=int)
    data = np.concatenate([bits, np.zeros(r, dtype=int)])
    for i in range(len(bits)):
        if data[i] == 1:
            data[i:i + len(poly)] ^= poly
    return np.all(data[-r:] == 0)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.info_mask = 1 - self.frozen_bits
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.info_mask == 1)[0]

    def decode(self, llr_ch):
        llr_channel = channel_llr_to_decoder(llr_ch)
        N = self.N
        n = self.n
        L = self.list_size

        llrs_list = []
        bits_list = []
        for _ in range(L):
            llrs = np.full((n + 1, N), -INF, dtype=np.float64)
            llrs[n, :] = llr_channel
            llrs_list.append(llrs)
            bits_list.append(np.full((n + 1, N), -1, dtype=np.int8))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0
        dm = np.zeros(L, dtype=np.float64)

        for i in range(N):
            if self.info_mask[i] == 0:
                for dd in range(L):
                    llrs_list[dd][0, i] = _li(0, i, llrs_list[dd], bits_list[dd])
                    bits_list[dd][0, i] = 0
                    if llrs_list[dd][0, i] < 0:
                        pm[dd] += abs(llrs_list[dd][0, i])
            else:
                for dd in range(L):
                    llrs_list[dd][0, i] = _li(0, i, llrs_list[dd], bits_list[dd])
                    bits_list[dd][0, i] = 1 if llrs_list[dd][0, i] < 0 else 0
                    dm[dd] = abs(llrs_list[dd][0, i])

                if L > 1:
                    pm_dm = np.empty(2 * L, dtype=np.float64)
                    pm_dm[:L] = pm
                    pm_dm[L:] = pm + dm
                    idx_sort = np.argsort(pm_dm)

                    idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                    idx_min_up = idx_sort[L:][idx_sort[L:] < L]
                    for bb in range(len(idx_min_low)):
                        low = idx_min_low[bb]
                        up = idx_min_up[bb]
                        llrs_list[up] = llrs_list[low].copy()
                        bits_list[up] = bits_list[low].copy()
                        bits_list[up][0, i] = 1 - bits_list[low][0, i]
                        pm[up] = pm_dm[low + L]

        candidates = list(range(L))
        if self.crc_length > 0:
            crc_ok = []
            for dd in candidates:
                u_hat = bits_list[dd][0, :]
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_ok.append(dd)
            if crc_ok:
                candidates = crc_ok

        best = candidates[np.argmin(pm[candidates])]
        u_hat = bits_list[best][0, :].astype(int)
        return u_hat, pm[best]
