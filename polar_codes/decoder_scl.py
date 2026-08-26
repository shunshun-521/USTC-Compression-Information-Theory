"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _PSCD, _prepare_channel_llr, _bit_reversed, precompute_sc_indices


CRC_POLYS = {
    8: 0x107,
    16: 0x11021,
}


def _crc_division(info_bits, crc_length):
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    remainder = _crc_division(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（基于 PSCD 路径扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])

        if crc_length > 0:
            info_natural = np.where(~self.frozen_bits)[0]
            self.crc_positions = info_natural[-crc_length:]
            self.crc_info_positions = info_natural[:-crc_length]
        else:
            self.crc_positions = np.array([], dtype=int)
            self.crc_info_positions = np.array([], dtype=int)

        self.layer_offset, self.llr_layer_vec, self.bit_layer_vec, _ = precompute_sc_indices(N)

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if hard == bit else abs(llr_val)

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        paths = [(0.0, _PSCD(self.N, self.frozen_bits))]
        paths[0][1].L[:, 0] = llr
        u_hat_paths = [np.zeros(self.N, dtype=np.int8)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_candidates = []

            for pidx, (pm, dec) in enumerate(paths):
                dec._update_llrs(l)
                llr0 = dec.L[l, self.n]
                u_prev = u_hat_paths[pidx].copy()

                if l in self.frozen_set:
                    bit = 0
                    new_pm = pm + self._pm_penalty(llr0, bit)
                    dec_copy = copy.deepcopy(dec)
                    dec_copy.B[l, self.n] = bit
                    dec_copy._update_bits(l)
                    u_prev[l] = bit
                    new_candidates.append((new_pm, dec_copy, u_prev))
                else:
                    for bit in (0, 1):
                        dec_copy = copy.deepcopy(dec)
                        new_pm = pm + self._pm_penalty(llr0, bit)
                        dec_copy.B[l, self.n] = bit
                        dec_copy._update_bits(l)
                        u_new = u_prev.copy()
                        u_new[l] = bit
                        new_candidates.append((new_pm, dec_copy, u_new))

            new_candidates.sort(key=lambda x: x[0])
            new_candidates = new_candidates[: self.list_size]
            paths = [(pm, dec) for pm, dec, _ in new_candidates]
            u_hat_paths = [u for _, _, u in new_candidates]

        if self.crc_length > 0:
            valid = []
            for pm, u in zip([p[0] for p in paths], u_hat_paths):
                info_bits = u[self.crc_info_positions]
                payload = np.concatenate([info_bits, u[self.crc_positions]])
                if crc_check(payload, self.crc_length):
                    valid.append((pm, u))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1].astype(int), valid[0][0]

        best_idx = int(np.argmin([p[0] for p in paths]))
        return u_hat_paths[best_idx].astype(int), paths[best_idx][0]
