"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation
from encoder import polar_encode


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(data_bits, poly, crc_length):
    """MSB-first 串行 CRC。"""
    crc = 0
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    for bit in data_bits:
        if ((crc >> (crc_length - 1)) ^ int(bit)) & 1:
            crc = ((crc << 1) ^ poly) & mask
        else:
            crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_bits(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_bits(bits, poly, crc_length) == 0


def _path_llr_at_phi(llr_ch, u_hat, phi, frozen_bits):
    """计算已知前缀 u_hat[0:phi] 时第 phi 个比特的 LLR。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    found = [None]

    def decode_rec(llr_vec, stage, bit_idx):
        if found[0] is not None:
            return
        if stage == 0:
            if bit_idx == phi:
                found[0] = llr_vec[0]
            return

        half = 1 << (stage - 1)
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])

        if bit_idx + half <= phi:
            decode_rec(llr_left, stage - 1, bit_idx)
            if found[0] is not None:
                return
            u_left = u_hat[bit_idx:bit_idx + half]
            v_left = polar_encode(u_left)
            llr_right = g_operation(llr_vec[:half], llr_vec[half:], v_left)
            decode_rec(llr_right, stage - 1, bit_idx + half)
        elif bit_idx <= phi < bit_idx + half:
            decode_rec(llr_left, stage - 1, bit_idx)
        else:
            decode_rec(llr_left, stage - 1, bit_idx)
            if found[0] is not None:
                return
            u_left = u_hat[bit_idx:bit_idx + half]
            v_left = polar_encode(u_left)
            llr_right = g_operation(llr_vec[:half], llr_vec[half:], v_left)
            decode_rec(llr_right, stage - 1, bit_idx + half)

    decode_rec(llr_ch.copy(), n, 0)
    return found[0]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{"pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            candidates = []
            llr_cache = {}
            for path in paths:
                key = path["u_hat"][:phi].tobytes()
                if key not in llr_cache:
                    llr_cache[key] = _path_llr_at_phi(
                        llr_ch, path["u_hat"], phi, self.frozen_bits
                    )
                llr = llr_cache[key]

                if self.frozen_bits[phi]:
                    new_path = {
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][phi] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "pm": path["pm"] + self._pm_penalty(llr, bit),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda item: item["pm"])
            paths = candidates[: self.list_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p["pm"])

        if self.crc_length > 0:
            for path in paths:
                payload = path["u_hat"][self.info_positions]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_any
        return chosen["u_hat"].copy(), chosen["pm"]


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    sigma = eb_n0_to_sigma(8.0, 0.5)

    mismatches = 0
    for seed in range(20):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    print(f"SCL L=1 vs SC mismatches: {mismatches}/20")
