"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import copy

from decoder_sc import f_operation, g_operation, sc_decode_recursive


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    poly = _crc_poly(crc_length)
    reg = 0
    info_bits = np.asarray(info_bits, dtype=int)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _partial_sums(decisions, indices):
    """子树返回的部分和（与 SC 递归返回值一致）"""
    if len(indices) == 1:
        return [decisions[indices[0]]]
    half = len(indices) // 2
    left = _partial_sums(decisions, indices[:half])
    right = _partial_sums(decisions, indices[half:])
    return [left[i] ^ right[i] for i in range(half)] + right


def bit_llr_at_phi(llrs, decisions, phi):
    """计算比特 phi 的 LLR（已知 decisions[0:phi]）"""
    N = len(llrs)
    indices = list(range(N))

    def recurse(llr_list, idxs, target_phi):
        if len(idxs) == 1:
            return llr_list[0]
        half = len(idxs) // 2
        left_idx = idxs[:half]
        right_idx = idxs[half:]
        left_llr = [
            f_operation(llr_list[i], llr_list[i + half]) for i in range(half)
        ]
        enc_left = _partial_sums(decisions, left_idx)
        if target_phi in left_idx:
            return recurse(left_llr, left_idx, target_phi)
        right_llr = [
            g_operation(llr_list[i], llr_list[i + half], enc_left[i])
            for i in range(half)
        ]
        return recurse(right_llr, right_idx, target_phi)

    return float(recurse(llrs, indices, phi))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = max(1, int(list_size))
        self.crc_length = int(crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        if self.L == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits.astype(int))
            return u, 0.0

        llrs = llr_ch.tolist()
        paths = [{"pm": 0.0, "dec": [0] * N}]

        for phi in range(N):
            candidates = []
            for path in paths:
                llr = bit_llr_at_phi(llrs, path["dec"], phi)
                if self.frozen_bits[phi]:
                    pen = abs(llr) if llr < 0 else 0.0
                    candidates.append((path["pm"] + pen, path, 0))
                else:
                    for bit in (0, 1):
                        pen = 0.0 if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0) else abs(llr)
                        candidates.append((path["pm"] + pen, path, bit))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, path, bit in candidates[: self.L]:
                npth = {"pm": pm, "dec": path["dec"].copy()}
                npth["dec"][phi] = bit
                new_paths.append(npth)
            paths = new_paths

        best = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_idx = np.where(~self.frozen_bits)[0]
            ok = [
                p
                for p in paths
                if crc_check(np.array(p["dec"])[info_idx], self.crc_length)
            ]
            if ok:
                best = min(ok, key=lambda p: p["pm"])

        u_hat = np.array(best["dec"], dtype=int)
        return u_hat, best["pm"]
