"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import cn_op, g_operation, sc_decode_recursive, LLR_MAX


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_update(reg, bit, poly, crc_length):
    reg ^= int(bit) << (crc_length - 1)
    for _ in range(8 if crc_length == 8 else 1):
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    if crc_length <= 0:
        return True
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, poly, crc_length)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Sionna 风格递归列表）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_ind = np.zeros(N, dtype=np.float64)
        self.frozen_ind[self.frozen_bits] = 1.0
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.cw_ind = np.arange(N)

    def _sort_paths(self):
        order = np.argsort(self.msg_pm)
        self.msg_pm = self.msg_pm[order]
        self.dec_pointer = self.dec_pointer[order]

    def _duplicate_paths(self):
        L = self.list_size
        low = self.dec_pointer[:L]
        up = self.dec_pointer[L:]
        self.msg_uhat[up] = self.msg_uhat[low]
        self.msg_llr[up] = self.msg_llr[low]
        self.msg_pm[L:] = self.msg_pm[:L]

    def _update_pm(self, bit_idx):
        llr_clip = np.clip(self.msg_llr[self.dec_pointer, 0, bit_idx], -LLR_MAX, LLR_MAX)
        u_hat = self.msg_uhat[self.dec_pointer, 0, bit_idx]
        self.msg_pm += np.log(1.0 + np.exp(-(1 - 2 * u_hat) * llr_clip))

    def _update_frozen_pm(self, cw_ind):
        stage = int(math.log2(len(cw_ind)))
        llr_in = self.msg_llr[self.dec_pointer, stage, cw_ind]
        llr_clip = np.clip(llr_in, -LLR_MAX, LLR_MAX)
        self.msg_pm += np.sum(np.log(1.0 + np.exp(-llr_clip)), axis=-1)

    def _scl_recursive(self, cw_ind):
        n = len(cw_ind)
        stage = int(math.log2(n))

        if n > 1:
            if np.all(self.frozen_ind[cw_ind] == 1):
                self._update_frozen_pm(cw_ind)
                return

            left = cw_ind[: n // 2]
            right = cw_ind[n // 2:]

            for p in self.dec_pointer:
                self.msg_llr[p, stage - 1, left] = cn_op(
                    self.msg_llr[p, stage, left],
                    self.msg_llr[p, stage, right],
                )

            self._scl_recursive(left)

            for p in self.dec_pointer:
                u_up = self.msg_uhat[p, stage - 1, left]
                self.msg_llr[p, stage - 1, right] = g_operation(
                    self.msg_llr[p, stage, left],
                    self.msg_llr[p, stage, right],
                    u_up,
                )

            self._scl_recursive(right)

            for p in self.dec_pointer:
                u_left_up = self.msg_uhat[p, stage - 1, left]
                u_right_up = self.msg_uhat[p, stage - 1, right]
                u_left = (u_left_up != u_right_up).astype(np.float64)
                self.msg_uhat[p, stage, cw_ind] = np.concatenate([u_left, u_right_up])

        else:
            bit_idx = cw_ind[0]
            if self.frozen_ind[bit_idx] == 1:
                self._update_frozen_pm(cw_ind)
            else:
                L = self.list_size
                base_ptr = self.dec_pointer[0]
                for ptr in self.dec_pointer:
                    self.msg_llr[ptr] = self.msg_llr[base_ptr]
                    self.msg_uhat[ptr] = self.msg_uhat[base_ptr]
                self.msg_uhat[self.dec_pointer[L:], 0, bit_idx] = 1.0
                self._update_pm(bit_idx)
                self._sort_paths()
                self._duplicate_paths()

    def decode(self, llr_ch):
        if self.list_size == 1:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        n = self.n
        N = self.N

        self.msg_uhat = np.zeros((2 * L, n + 1, N), dtype=np.float64)
        self.msg_llr = np.zeros((2 * L, n + 1, N), dtype=np.float64)
        self.msg_pm = np.zeros(2 * L, dtype=np.float64)
        self.msg_pm[1:L] = LLR_MAX
        self.msg_pm[L + 1:] = LLR_MAX
        self.dec_pointer = np.arange(2 * L, dtype=int)
        self.msg_llr[0, n, :] = llr_ch
        for i in range(1, 2 * L):
            self.msg_llr[i] = self.msg_llr[0]
            self.msg_uhat[i] = self.msg_uhat[0]

        self._scl_recursive(self.cw_ind)
        self._sort_paths()

        best_idx = self.dec_pointer[0]
        u_hat = self.msg_uhat[best_idx, 0, :].astype(int)

        if self.crc_length > 0:
            for idx in self.dec_pointer[:L]:
                cand = self.msg_uhat[idx, 0, :].astype(int)
                info_bits = cand[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    pos = np.where(self.dec_pointer == idx)[0][0]
                    pm = self.msg_pm[pos]
                    if best_idx is None or pm < self.msg_pm[0]:
                        u_hat = cand
                        best_idx = idx

        pm = float(self.msg_pm[0])
        return u_hat, pm
