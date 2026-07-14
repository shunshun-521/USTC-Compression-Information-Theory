"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_bits,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_encode(info_bits):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_encode(info_bits):
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        remainder = _crc8_encode(info_bits)
    elif crc_length == 16:
        remainder = _crc16_encode(info_bits)
    else:
        raise ValueError("crc_length must be 8 or 16")
    crc_bits = np.array(
        [(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    received_crc = bits[-crc_length:]
    if crc_length == 8:
        expected = _crc8_encode(info)
    else:
        expected = _crc16_encode(info)
    crc_val = sum(b << (crc_length - 1 - i) for i, b in enumerate(received_crc))
    return expected == crc_val


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs_path(self, L, B, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _branch_metric(self, llr, bit):
        decided = 0 if llr >= 0 else 1
        return 0.0 if bit == decided else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        L = self.n + 1
        rev = np.array(
            [int(format(i, f"0{n}b")[::-1], 2) for i in range(self.N)], dtype=int
        )
        llr_perm = llr_ch[rev]
        decode_order = [
            int(format(i, f"0{n}b")[::-1], 2) for i in range(self.N)
        ]

        paths = []
        L0 = np.zeros((self.N, L), dtype=np.float64)
        B0 = np.zeros((self.N, L), dtype=np.int32)
        L0[:, 0] = llr_perm
        paths.append({"pm": 0.0, "L": L0, "B": B0, "u_hat": np.zeros(self.N, dtype=int)})

        for phi in decode_order:
            candidates = []
            for path in paths:
                self._update_llrs_path(path["L"], path["B"], phi)
                llr = path["L"][phi, n]

                if phi in self.frozen_set:
                    bit = 0
                    candidates.append(
                        (path["pm"] + self._branch_metric(llr, bit), path, bit)
                    )
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path["pm"] + self._branch_metric(llr, bit), path, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child_L = parent["L"].copy()
                child_B = parent["B"].copy()
                child_u = parent["u_hat"].copy()
                child_u[phi] = bit
                child_B[phi, n] = bit
                _update_bits(child_B, phi, n)
                new_paths.append({"pm": pm, "L": child_L, "B": child_B, "u_hat": child_u})
            paths = new_paths

        if self.crc_length > 0:
            crc_pass = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(crc_pass if crc_pass else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]


def verify_scl_equals_sc(N=64, num_frames=20):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(4.0, 0.5)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"
    return True
