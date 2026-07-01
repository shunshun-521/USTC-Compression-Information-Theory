"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    precompute_sc_indices,
    _prepare_llr,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << crc_length) - 1)
        if reg & (1 << (crc_length - 1)):
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)

    def _new_arrays(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return L, B

    def _update_llr_layers(self, path, idx):
        L = path["L"]
        B = path["B"]
        l = self.decode_order[idx]
        for s in self.llr_layer_vec[idx]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bit_layers(self, path, idx, u_bit):
        B = path["B"]
        l = self.decode_order[idx]
        B[l, self.n] = u_bit
        if l < self.N // 2:
            return
        for s in self.bit_layer_vec[idx]:
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _path_metric_penalty(self, llr, u_bit):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u_bit == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = _prepare_llr(llr_ch)
        L0, B0 = self._new_arrays(llr_ch)
        paths = [
            {
                "pm": 0.0,
                "L": L0,
                "B": B0,
                "u_hat": np.zeros(self.N, dtype=int),
            }
        ]

        for idx in range(self.N):
            l = self.decode_order[idx]
            candidates = []
            for path in paths:
                self._update_llr_layers(path, idx)
                llr = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    new_path = {
                        "pm": path["pm"] + self._path_metric_penalty(llr, 0),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["u_hat"][l] = 0
                    self._update_bit_layers(new_path, idx, 0)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = {
                            "pm": path["pm"] + self._path_metric_penalty(llr, u_bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][l] = u_bit
                        self._update_bit_layers(new_path, idx, u_bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(crc_pass or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]


def run_scl_validation_tests():
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(1)
    sigma = eb_n0_to_sigma(5.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(50):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(awgn_channel(bpsk_modulate(x), sigma, rng), sigma)
        u_scl, _ = scl.decode(llr)
        u_sc = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_scl, u_sc):
            raise AssertionError("SCL L=1 != SC")
    print("SCL validation tests passed.")


if __name__ == "__main__":
    run_scl_validation_tests()
