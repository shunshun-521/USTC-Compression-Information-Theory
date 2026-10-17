"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _li


def _crc_poly(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
    if crc_length == 16:
        return np.array(
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
        )
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    r = crc_length
    reg = np.zeros(r, dtype=np.int8)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= poly[1:]
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length

    def decode(self, llr_ch):
        L = self.list_size
        N = self.N
        n = self.n
        llr_ch = llr_ch.astype(np.float64)

        llrs = []
        bits = []
        for _ in range(L):
            llr_arr = -np.inf * np.ones((n + 1, N), dtype=np.float64)
            llr_arr[-1, :] = llr_ch
            llrs.append(llr_arr)
            bits.append(-np.ones((n + 1, N), dtype=np.int8))

        pm = np.full(L, np.inf, dtype=np.float64)
        pm[0] = 0.0

        for i in range(N):
            dm = np.zeros(L, dtype=np.float64)
            for path in range(L):
                if pm[path] == np.inf:
                    continue
                llr_i = _li(0, i, llrs[path], bits[path])
                llrs[path][0, i] = llr_i
                if self.frozen_bits[i]:
                    bits[path][0, i] = 0
                    pm[path] += max(0.0, -llr_i) if llr_i < 0 else 0.0
                else:
                    bits[path][0, i] = 1 if llr_i < 0 else 0
                    dm[path] = abs(llr_i)

            if self.frozen_bits[i] or L == 1:
                continue

            pm_dm = np.concatenate([pm, pm + dm])
            idx_sort = np.argsort(pm_dm)
            idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
            idx_min_up = idx_sort[L:][idx_sort[L:] < L]

            for low_idx, up_idx in zip(idx_min_low, idx_min_up):
                llrs[up_idx] = np.copy(llrs[low_idx])
                bits[up_idx] = np.copy(bits[low_idx])
                bits[up_idx][0, i] = 1 - bits[low_idx][0, i]
                pm[up_idx] = pm_dm[low_idx + L]

            for k in range(L):
                if idx_sort[k] < L:
                    pm[idx_sort[k]] = pm_dm[idx_sort[k]]

        paths = [(pm[p], bits[p][0, :].astype(np.int8)) for p in range(L) if pm[p] < np.inf]

        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            valid = []
            for pm_val, u_hat in paths:
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append((pm_val, u_hat))
            if valid:
                paths = valid

        paths.sort(key=lambda x: x[0])
        return paths[0][1], float(paths[0][0])
