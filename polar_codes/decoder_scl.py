"""
极化码 SCL 译码器（含 CRC 辅助 CA-SCL）
L=1 时等价于 SC；L>1 时采用逐信息位分裂 + 路径度量裁剪
"""
import copy
import numpy as np
from decoder_sc import sc_decode, _align_llr_to_generator, _sc_tree_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _path_metric(llr_segment, bit_segment):
    pm = 0.0
    for llr, u in zip(llr_segment, bit_segment):
        u_llr = 0 if llr >= 0 else 1
        if u != u_llr:
            pm += abs(llr)
    return pm


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None, bit_reverse_codeword=True):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices if info_indices is not None else np.where(~self.frozen_bits)[0])
        self.bit_reverse_codeword = bit_reverse_codeword

    def decode(self, llr_ch):
        llr = _align_llr_to_generator(np.asarray(llr_ch, dtype=np.float64), self.bit_reverse_codeword)
        if self.list_size == 1:
            u = sc_decode(
                llr_ch,
                self.frozen_bits,
                bit_reverse_codeword=self.bit_reverse_codeword,
                info_indices=self.info_indices,
            )
            return u, 0.0

        info_positions = list(self.info_indices)
        # 路径：(u_hat, pm)
        paths = [(np.zeros(self.N, dtype=np.int8), 0.0)]

        for phi in range(self.N):
            new_paths = []
            for u_hat, pm in paths:
                if self.frozen_bits[phi]:
                    u_hat = u_hat.copy()
                    u_hat[phi] = 0
                    llr_phi = llr[phi]
                    new_paths.append((u_hat, pm + (abs(llr_phi) if llr_phi < 0 else 0.0)))
                    continue

                if phi not in info_positions:
                    continue

                for u_bit in (0, 1):
                    cand = u_hat.copy()
                    cand[phi] = u_bit
                    llr_phi = llr[phi]
                    pen = 0.0 if (0 if llr_phi >= 0 else 1) == u_bit else abs(llr_phi)
                    new_paths.append((cand, pm + pen))

            if not new_paths:
                new_paths = paths
            new_paths.sort(key=lambda x: x[1])
            paths = new_paths[: self.list_size]

        # 对非信息位用 SC 补全
        finalized = []
        for u_hat, pm in paths:
            full = sc_decode(
                llr_ch,
                self.frozen_bits,
                bit_reverse_codeword=self.bit_reverse_codeword,
                info_indices=self.info_indices,
            )
            # 保留候选信息位
            full[self.info_indices] = u_hat[self.info_indices]
            finalized.append((full, pm))

        if self.crc_length > 0:
            ok = [(u, p) for u, p in finalized if crc_check(u[self.info_indices], self.crc_length)]
            if ok:
                finalized = ok
        u_hat, pm = min(finalized, key=lambda x: x[1])
        return u_hat.astype(int), pm
