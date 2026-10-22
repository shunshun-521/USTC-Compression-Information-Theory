"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u):
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


def compute_llr_at_phi(llr_ch, u_hat, phi):
    """计算第 phi 个比特的 LLR（已知 u_hat[0:phi]）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    indices = list(range(N))

    def traverse(llrs, idxs):
        if len(idxs) == 1:
            i = idxs[0]
            if i < phi:
                return [int(u_hat[i])]
            if i == phi:
                return float(llrs[0])
            return [0]

        half = len(idxs) // 2
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        left_res = traverse(left_llr, idxs[:half])
        if isinstance(left_res, float):
            return left_res
        right_llr = np.array(
            [g_operation(llrs[i], llrs[i + half], left_res[i]) for i in range(half)],
            dtype=np.float64,
        )
        right_res = traverse(right_llr, idxs[half:])
        if isinstance(right_res, float):
            return right_res
        return [left_res[i] ^ right_res[i] for i in range(half)] + right_res

    res = traverse(llr_ch, indices)
    return float(res)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：仅在分裂时复制 u_hat 与 pm）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [(0.0, np.zeros(self.N, dtype=np.int8))]

        for phi in range(self.N):
            new_paths = []
            for pm, u_hat in paths:
                llr = compute_llr_at_phi(llr_ch, u_hat, phi)
                if self.frozen_bits[phi]:
                    pen = _pm_penalty(llr, 0)
                    u_new = u_hat.copy()
                    u_new[phi] = 0
                    new_paths.append((pm + pen, u_new))
                else:
                    for u in (0, 1):
                        u_new = u_hat.copy()
                        u_new[phi] = u
                        new_paths.append((pm + _pm_penalty(llr, u), u_new))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p[1][self.info_positions], self.crc_length)
            ]
            if valid:
                paths = valid

        pm, u_hat = min(paths, key=lambda x: x[0])
        return u_hat.copy(), pm


def scl_equivalent_sc_check(N=64, K=32):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)
    scl = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    mism = 0
    for _ in range(20):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(
            bpsk_modulate(polar_encode(u)) + np.random.normal(0, sigma, N), sigma
        )
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode_recursive(llr, frozen)
        if not np.array_equal(u_scl, u_sc):
            mism += 1
    return mism
