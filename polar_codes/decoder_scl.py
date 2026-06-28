"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    _bit_reversed,
    _sc_decode_core,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & mask
        if int(bit):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    if crc_length not in (8, 16):
        raise ValueError("crc_length must be 8 or 16")
    combined = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=int)])
    expected = _crc_remainder(combined, poly, crc_length)
    received = 0
    for i in range(crc_length):
        received = (received << 1) | int(bits[-crc_length + i])
    return expected == received


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_positions = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _update_llrs(self, L, B, l):
        n = self.n
        N = self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        N = self.N
        n = self.n
        llr_ch = llr_ch.astype(np.float64)

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        decode_order = [_bit_reversed(i, n) for i in range(N)]

        for l in decode_order:
            new_paths = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                self._update_llrs(L, B, l)
                llr = L[l, n]

                if l in self.frozen_set:
                    new_p = {
                        "L": L.copy(),
                        "B": B.copy(),
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                    }
                    new_p["B"][l, n] = 0
                    self._update_bits(new_p["B"], l)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = {
                            "L": L.copy(),
                            "B": B.copy(),
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                        }
                        new_p["B"][l, n] = u
                        self._update_bits(new_p["B"], l)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc_pm = None
        best_crc_u = None
        best_pm = paths[0]["pm"]
        best_u = paths[0]["B"][:, n].astype(int)

        for path in paths:
            u_hat = path["B"][:, n].astype(int)
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = u_hat
            if self.crc_length > 0:
                info_bits = u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    if best_crc_pm is None or path["pm"] < best_crc_pm:
                        best_crc_pm = path["pm"]
                        best_crc_u = u_hat

        return (best_crc_u if best_crc_u is not None else best_u), (
            best_crc_pm if best_crc_pm is not None else best_pm
        )


def validate_scl_equals_sc():
    """单路径 SCL 应等价于 SC。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("SCL L=1 equals SC validation passed.")


if __name__ == "__main__":
    validate_scl_equals_sc()
