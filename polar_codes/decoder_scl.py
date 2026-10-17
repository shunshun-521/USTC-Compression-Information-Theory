"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_bp import BPDecoder

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
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int32)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int32)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class SCLDecoder:
    """SCL 译码器（基于 BP 后验 LLR 的位级列表搜索）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._bp = BPDecoder(N, frozen_bits, max_iter=1)

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=np.int32)}]

        for phi in range(self.N):
            fixed = np.full(self.N, -1, dtype=np.int32)
            new_paths = []
            for path in paths:
                if phi > 0:
                    fixed[:phi] = path["u"][:phi]
                post = self._bp.posterior_llr(llr_ch, fixed)
                llr0 = post[phi]

                if self.frozen_bits[phi]:
                    u_bit = 0
                    pm = path["pm"] + self._pm_penalty(llr0, u_bit)
                    u = path["u"].copy()
                    u[phi] = u_bit
                    new_paths.append({"pm": pm, "u": u})
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr0, u_bit)
                        u = path["u"].copy()
                        u[phi] = u_bit
                        new_paths.append({"pm": pm, "u": u})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            chosen = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            chosen = min(paths, key=lambda p: p["pm"])

        return chosen["u"].copy(), chosen["pm"]


def verify_scl_equals_sc(N=64, K=32, seed=1):
    """L=1 的 SCL 应与 SC 一致。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(3.0, K / N)
    rng = np.random.default_rng(seed)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(20):
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
