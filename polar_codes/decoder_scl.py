"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import f_operation, g_operation, _B_check, _s_updater, _Li


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径度量 + 列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        frozen_bits = np.asarray(frozen_bits)
        self.info_mask = np.ones(N, dtype=np.int8)
        if frozen_bits.dtype == bool:
            self.info_mask[frozen_bits] = 0
        else:
            self.info_mask[frozen_bits.astype(int) > 0] = 0
        self.br = bit_reversal_permutation(N)

    def _new_path(self, llr_ch):
        llrs = -np.inf * np.ones((self.n + 1, self.N), dtype=np.float64)
        llrs[self.n, :] = llr_ch
        s = -np.ones((self.n + 1, self.N), dtype=np.int8)
        return llrs, s, 0.0, np.zeros(self.N, dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        L = self.list_size
        paths = [self._new_path(llr_ch) for _ in range(L)]
        pms = np.full(L, np.inf, dtype=np.float64)
        pms[0] = 0.0
        active = 1

        for ii in range(self.N):
            candidates = []
            for l in range(active):
                llrs, s, pm, u_hat = paths[l]
                llrs = llrs.copy()
                s = s.copy()
                u_hat = u_hat.copy()
                ll_val = _Li(0, ii, llrs, s, self.n)

                if self.info_mask[ii] == 0:
                    pen = 0.0 if ll_val >= 0 else abs(ll_val)
                    u_bit = 0
                    s[0, ii] = 0
                    llrs[0, ii] = np.inf
                    u_hat[ii] = 0
                    candidates.append((pm + pen, llrs, s, u_hat))
                else:
                    for u_bit in (0, 1):
                        llrs_c = llrs.copy()
                        s_c = s.copy()
                        u_c = u_hat.copy()
                        s_c[0, ii] = u_bit
                        llrs_c[0, ii] = ll_val
                        u_c[ii] = u_bit
                        consistent = (u_bit == 0 and ll_val >= 0) or (u_bit == 1 and ll_val < 0)
                        pen = 0.0 if consistent else abs(ll_val)
                        candidates.append((pm + pen, llrs_c, s_c, u_c))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L]
            active = len(candidates)
            paths = [(c[1], c[2], c[0], c[3]) for c in candidates]
            pms = np.array([c[0] for c in candidates] + [np.inf] * (L - active))

        best_idx = 0
        if self.crc_length > 0:
            info_idx = np.where(self.info_mask > 0)[0]
            valid = []
            for i, (_, _, pm, u_hat) in enumerate(paths[:active]):
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append((pm, i))
            if valid:
                best_idx = min(valid, key=lambda x: x[0])[1]
            else:
                best_idx = int(np.argmin(pms))
        else:
            best_idx = int(np.argmin(pms))

        u_hat = paths[best_idx][3]
        return u_hat, float(pms[best_idx])
