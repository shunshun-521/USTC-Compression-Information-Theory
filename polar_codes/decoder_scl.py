"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _frozen_to_info_mask, _li

try:
    from numba import jit

    _NUMBA = True
except ImportError:  # pragma: no cover
    _NUMBA = False

    def jit(*_args, **_kwargs):
        def wrapper(fn):
            return fn

        return wrapper


_CRC8_POLY = np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=np.int32)


@jit(nopython=True)
def _crc_calculator(x, crc_polynomial):
    crc_len = len(crc_polynomial)
    x_new = np.zeros(len(x) + crc_len - 1, dtype=np.int32)
    x_new[: len(x)] = x
    for ii in range(len(x_new) - crc_len + 1):
        if x_new[ii] == 1:
            x_new[ii : ii + crc_len] ^= crc_polynomial
    return x_new[-crc_len + 1 :]


@jit(nopython=True)
def _scl_decode_core(llr_channel, if_information_bit, list_size):
    n = int(math.log2(len(llr_channel)))
    N = 1 << n
    L = list_size

    llrs = [-np.inf * np.ones((n + 1, N), dtype=np.float64) for _ in range(L)]
    for dd in range(L):
        llrs[dd][-1, :] = llr_channel

    s = [-1 * np.ones((n + 1, N), dtype=np.int8) for _ in range(L)]
    dm = np.zeros(L, dtype=np.float64)
    pm = np.inf * np.ones(L, dtype=np.float64)
    pm[0] = 0.0
    pm_dm = np.zeros(2 * L, dtype=np.float64)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 0
                pm[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                dm[dd] = abs(llrs[dd][0, ii])

        if if_information_bit[ii] != 0 and L > 1:
            pm_dm[:L] = pm
            pm_dm[L:] = pm + dm
            idx_sort = np.argsort(pm_dm)
            idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
            idx_min_up = idx_sort[L:][idx_sort[L:] < L]
            for bb in range(len(idx_min_low)):
                llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])
                s[idx_min_up[bb]] = np.copy(s[idx_min_low[bb]])
                s[idx_min_up[bb]][0, ii] = 1 - s[idx_min_low[bb]][0, ii]
                pm[idx_min_up[bb]] = pm_dm[idx_min_low[bb] + L]

    best = int(np.argmin(pm))
    return s[best][0, :], pm[best]


@jit(nopython=True)
def _scl_crc_decode_core(
    llr_channel, if_information_bit, information_indices, k_crc, list_size, crc_polynomial
):
    u_hat, pm = _scl_decode_core(llr_channel, if_information_bit, list_size)
    # 为 CRC 校验，需要所有路径 — 简化：复用 SCL 再对 L 条路径扩展
    # 使用与 _scl_decode_core 相同循环但在末尾筛选 CRC
    n = int(math.log2(len(llr_channel)))
    N = 1 << n
    L = list_size

    llrs = [-np.inf * np.ones((n + 1, N), dtype=np.float64) for _ in range(L)]
    for dd in range(L):
        llrs[dd][-1, :] = llr_channel
    s = [-1 * np.ones((n + 1, N), dtype=np.int8) for _ in range(L)]
    dm = np.zeros(L, dtype=np.float64)
    pm = np.inf * np.ones(L, dtype=np.float64)
    pm[0] = 0.0
    pm_dm = np.zeros(2 * L, dtype=np.float64)

    for ii in range(N):
        if if_information_bit[ii] == 0:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 0
                pm[dd] += -llrs[dd][0, ii] * (llrs[dd][0, ii] < 0)
        else:
            for dd in range(L):
                llrs[dd][0, ii] = _li(0, ii, llrs[dd], s[dd])
                s[dd][0, ii] = 1 if llrs[dd][0, ii] < 0 else 0
                dm[dd] = abs(llrs[dd][0, ii])

        if if_information_bit[ii] != 0 and L > 1:
            pm_dm[:L] = pm
            pm_dm[L:] = pm + dm
            idx_sort = np.argsort(pm_dm)
            idx_min_low = idx_sort[:L][idx_sort[:L] >= L] - L
            idx_min_up = idx_sort[L:][idx_sort[L:] < L]
            for bb in range(len(idx_min_low)):
                llrs[idx_min_up[bb]] = np.copy(llrs[idx_min_low[bb]])
                s[idx_min_up[bb]] = np.copy(s[idx_min_low[bb]])
                s[idx_min_up[bb]][0, ii] = 1 - s[idx_min_low[bb]][0, ii]
                pm[idx_min_up[bb]] = pm_dm[idx_min_low[bb] + L]

    crc_ok = np.zeros(L, dtype=np.int8)
    crc_len = len(crc_polynomial)
    for dd in range(L):
        payload = s[dd][0, information_indices[:k_crc]]
        crc_calc = _crc_calculator(payload, crc_polynomial)
        crc_ref = s[dd][0, information_indices[k_crc : k_crc + crc_len]]
        if np.all(crc_calc == crc_ref):
            crc_ok[dd] = 1

    n_ok = int(np.sum(crc_ok))
    if n_ok == 1:
        dd_best = int(np.nonzero(crc_ok == 1)[0][0])
    elif n_ok > 1:
        cands = np.nonzero(crc_ok == 1)[0]
        dd_best = int(cands[np.argmin(pm[cands])])
    else:
        dd_best = int(np.argmin(pm))
    return s[dd_best][0, :], pm[dd_best]


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 编码。"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = _CRC8_POLY if crc_length == 8 else np.array(
        [int(b) for b in format(0x8005, "016b")], dtype=np.int32
    )
    rem = _crc_calculator(info_bits, poly[:crc_length])
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int32)
    poly = _CRC8_POLY if crc_length == 8 else np.array(
        [int(b) for b in format(0x8005, "016b")], dtype=np.int32
    )
    if len(bits) < crc_length:
        return False
    rem = _crc_calculator(bits, poly[:crc_length])
    return np.all(rem == bits[-crc_length:])


class SCLDecoder:
    """SCL / CA-SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0].astype(np.int32)
        self.if_info = _frozen_to_info_mask(self.frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.crc_length > 0:
            k_crc = len(self.info_positions) - self.crc_length
            u_hat, pm = _scl_crc_decode_core(
                llr_ch,
                self.if_info,
                self.info_positions,
                k_crc,
                self.list_size,
                _CRC8_POLY,
            )
        else:
            u_hat, pm = _scl_decode_core(llr_ch, self.if_info, self.list_size)
        return u_hat.astype(int), float(pm)
