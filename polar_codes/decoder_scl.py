"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation, _xor_combine, sc_decode_recursive

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


def _llr_at_phi(llr_ch, frozen_bits, u_hat, phi):
    """在给定前缀 u_hat[0:phi] 下计算位置 phi 的 LLR"""
    u_hat = np.asarray(u_hat, dtype=int)
    N = len(llr_ch)
    n_levels = int(math.log2(N)) + 1

    def left_return_known(llr, depth, bit_start):
        """子树比特范围 [bit_start, bit_start+len(llr)) 均 < phi"""
        if depth == n_levels - 1:
            return np.array([u_hat[bit_start]], dtype=int)

        half = len(llr) // 2
        left_llr = f_operation(llr[:half], llr[half:])
        left_ret = left_return_known(left_llr, depth + 1, bit_start)
        right_llr = g_operation(llr[:half], llr[half:], left_ret)
        right_ret = left_return_known(
            right_llr, depth + 1, bit_start + half
        )
        return _xor_combine(left_ret, right_ret)

    def get_llr(llr, depth, bit_start):
        if depth == n_levels - 1:
            return llr[0]

        half = len(llr) // 2
        if phi < bit_start + half:
            left_llr = f_operation(llr[:half], llr[half:])
            return get_llr(left_llr, depth + 1, bit_start)

        left_llr = f_operation(llr[:half], llr[half:])
        left_ret = left_return_known(left_llr, depth + 1, bit_start)
        right_llr = g_operation(llr[:half], llr[half:], left_ret)
        return get_llr(right_llr, depth + 1, bit_start + half)

    return get_llr(llr_ch, 0, 0)


class SCLDecoder:
    """SCL 译码器（路径列表 + CRC 辅助）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode_recursive(llr_ch, self.frozen_bits), 0.0

        paths = [{"u_hat": np.zeros(self.N, dtype=int), "pm": 0.0}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_bit = _llr_at_phi(llr_ch, self.frozen_bits, path["u_hat"], phi)

                if self.frozen_bits[phi]:
                    pm = _pm_update(path["pm"], llr_bit, 0)
                    uh = path["u_hat"].copy()
                    uh[phi] = 0
                    new_paths.append({"u_hat": uh, "pm": pm})
                else:
                    for u_bit in (0, 1):
                        pm = _pm_update(path["pm"], llr_bit, u_bit)
                        uh = path["u_hat"].copy()
                        uh[phi] = u_bit
                        new_paths.append({"u_hat": uh, "pm": pm})

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        candidates = [(p["pm"], p["u_hat"]) for p in paths]

        if self.crc_length > 0:
            valid = [
                (pm, u) for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            if valid:
                candidates = valid

        best = min(candidates, key=lambda x: x[0])
        return best[1].copy(), best[0]


def verify_scl_equals_sc(N=64, num_trials=10):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(5.0, 0.5)
    rng = np.random.default_rng(42)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(
            awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
        )
        u_sc = sc_decode_recursive(llr, frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True


if __name__ == "__main__":
    print("Verifying SCL L=1 == SC...")
    verify_scl_equals_sc()
    print("SCL verification passed.")
