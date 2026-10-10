"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
from decoder_sc import _Li, frozen_bits_to_info_mask

try:
    from numba import jit

    _NUMBA = True
except ImportError:

    def jit(*args, **kwargs):
        def wrap(fn):
            return fn

        return wrap


@jit(nopython=True)
def _scl_decode_core(llr_channel, if_information_bit, L):
    N = len(llr_channel)
    n = int(np.log2(N))
    llrs = [-np.inf * np.ones((n + 1, 1 << n), dtype=np.float32) for _ in range(L)]
    for dd in range(L):
        llrs[dd][-1, :] = llr_channel
    s = [-1 * np.ones((n + 1, 1 << n), dtype=np.int8) for _ in range(L)]
    DM = np.zeros(L)
    PM = np.inf * np.ones(L, dtype=np.float32)
    PM[0] = 0.0
    PM_DM = np.zeros(2 * L)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            for dd in range(L):
                llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 0
                PM[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                DM[dd] = np.abs(llrs[dd][0, ii])

        if if_information_bit[ii] and L > 1:
            PM_DM[:L] = PM
            PM_DM[L:] = PM + DM
            idx_sort = np.argsort(PM_DM)
            idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
            idx_min_up = idx_sort[L:][idx_sort[L:] < L]
            len_list_change = len(idx_min_low)
            if len_list_change != 0:
                for bb in range(len_list_change):
                    llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])
                    s[idx_min_up[bb]] = np.copy(s[idx_min_low[bb]])
                    s[idx_min_up[bb]][0, ii] = 1 - s[idx_min_low[bb]][0, ii]
                PM[idx_min_up] = PM_DM[idx_min_low + L]

    dd_best = int(np.argmin(PM))
    return s[dd_best][0, :], PM[dd_best]


_CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
_CRC16_POLY = np.array(
    [1] + [0] * 12 + [1, 0, 0, 0, 1], dtype=np.int8
)  # 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    r = len(poly) - 1
    msg = np.concatenate([np.asarray(info_bits, dtype=np.int8), np.zeros(r, dtype=np.int8)])
    for i in range(len(info_bits)):
        if msg[i] == 1:
            msg[i : i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[len(info_bits) :]])


def crc_check(bits, crc_length=8):
    poly = _crc_poly(crc_length)
    r = crc_length
    data = np.asarray(bits, dtype=np.int8)
    msg = np.concatenate([data[:-r], np.zeros(r, dtype=np.int8)])
    msg[: len(data) - r] = data[:-r]
    rem = crc_encode(data[:-r], crc_length)[-r:]
    return np.array_equal(rem, data[-r:])


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int8)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_mask = frozen_bits_to_info_mask(self.frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float32)
        L = self.list_size
        if self.crc_length > 0:
            return self._decode_crc(llr_ch, L)
        u_hat, pm = _scl_decode_core(llr_ch, self.info_mask, L)
        return u_hat.astype(int), float(pm)

    def _decode_crc(self, llr_ch, L):
        """多路径译码后按 CRC 筛选"""
        N = self.N
        n = int(np.log2(N))
        llrs = [-np.inf * np.ones((n + 1, 1 << n), dtype=np.float32) for _ in range(L)]
        for dd in range(L):
            llrs[dd][-1, :] = llr_ch
        s = [-1 * np.ones((n + 1, 1 << n), dtype=np.int8) for _ in range(L)]
        DM = np.zeros(L)
        PM = np.inf * np.ones(L, dtype=np.float32)
        PM[0] = 0.0
        PM_DM = np.zeros(2 * L)
        info_idx = np.where(self.info_mask > 0)[0]

        for ii in range(N):
            if self.info_mask[ii] == 0:
                for dd in range(L):
                    llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                    s[dd][0, ii] = 0
                    PM[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
            else:
                for dd in range(L):
                    llrs[dd][0, ii] = _Li(0, ii, llrs[dd], s[dd])
                    s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                    DM[dd] = np.abs(llrs[dd][0, ii])

            if self.info_mask[ii] and L > 1:
                PM_DM[:L] = PM
                PM_DM[L:] = PM + DM
                idx_sort = np.argsort(PM_DM)
                idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
                idx_min_up = idx_sort[L:][idx_sort[L:] < L]
                if len(idx_min_low) > 0:
                    for bb in range(len(idx_min_low)):
                        llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])
                        s[idx_min_up[bb]] = np.copy(s[idx_min_low[bb]])
                        s[idx_min_up[bb]][0, ii] = 1 - s[idx_min_low[bb]][0, ii]
                    PM[idx_min_up] = PM_DM[idx_min_low + L]

        candidates = []
        for dd in range(L):
            u = s[dd][0, :].astype(int)
            bits = u[info_idx]
            if crc_check(bits, self.crc_length):
                candidates.append((PM[dd], u))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1], float(candidates[0][0])
        dd_best = int(np.argmin(PM))
        return s[dd_best][0, :].astype(int), float(PM[dd_best])
