"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    sc_decode,
)
from encoder import bit_reversal_permutation

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length <= 8 else 1):
            if crc_length <= 8:
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
            else:
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        rem = _crc_remainder(info_bits, _CRC8_POLY, 8)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=int)
    elif crc_length == 16:
        rem = _crc_remainder(info_bits, _CRC16_POLY, 16)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        poly, mask = _CRC8_POLY, (1 << 8) - 1
    elif crc_length == 16:
        poly, mask = _CRC16_POLY, 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _pm_penalty(llr, u):
    """路径度量惩罚：比特与 LLR 符号不一致时加 |LLR|。"""
    u_hard = 0 if llr >= 0 else 1
    return 0.0 if u == u_hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.brp = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n, list_sz = self.N, self.n, self.list_size

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.float64),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []

            for pidx, path in enumerate(paths):
                L_arr, B = path["L"], path["B"]
                _update_llrs(l, L_arr, B, n, N)
                llr_root = L_arr[l, n]

                if self.frozen_bits[phi]:
                    pen = _pm_penalty(llr_root, 0)
                    new_path = {
                        "pm": path["pm"] + pen,
                        "L": L_arr.copy(),
                        "B": B.copy(),
                        "u": path["u"].copy(),
                    }
                    new_path["u"][l] = 0
                    new_path["B"][l, n] = 0
                    _update_bits(l, new_path["B"], n, N)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        pen = _pm_penalty(llr_root, u_bit)
                        new_path = {
                            "pm": path["pm"] + pen,
                            "L": L_arr.copy(),
                            "B": B.copy(),
                            "u": path["u"].copy(),
                        }
                        new_path["u"][l] = u_bit
                        new_path["B"][l, n] = u_bit
                        _update_bits(l, new_path["B"], n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[:list_sz]

        brp = self.brp
        info_mask = ~self.frozen_bits

        best_path = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_nat = p["u"][brp]
                info_bits = u_nat[info_mask]
                if len(info_bits) >= self.crc_length and crc_check(
                    info_bits[-self.crc_length :], self.crc_length
                ):
                    # CRC 覆盖最后 r 位（含在信息位索引中）
                    payload = info_bits[: -self.crc_length]
                    encoded = crc_encode(payload, self.crc_length)
                    if np.array_equal(info_bits, encoded):
                        valid.append(p)
            if valid:
                valid.sort(key=lambda x: x["pm"])
                best_path = valid[0]

        u_hat = best_path["u"][brp]
        return u_hat, best_path["pm"]


def verify_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, K / N)
    rng = np.random.default_rng(7)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl)
