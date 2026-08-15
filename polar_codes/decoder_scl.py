"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _frozen_set_from_mask,
    _prepare_llr,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = _frozen_set_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        """与 LLR 符号不一致时加 |LLR| 惩罚。"""
        hard = 0 if llr >= 0 else 1
        if hard == bit:
            return 0.0
        return abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm（最优路径度量）
        """
        llr0 = _prepare_llr(llr_ch, self.N)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr0

        for phase in range(N):
            l = _bit_reversed_index(phase, n)
            new_paths = []

            for path in paths:
                L = path["L"]
                B = path["B"]

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = int(2 ** (s + 1))
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s],
                                L[j, s],
                                B[j - branch_size, s + 1],
                            )

                cur_llr = L[l, n]
                candidates = []

                if l in self.frozen_set:
                    bit = 0
                    pm = path["pm"] + self._path_metric_penalty(cur_llr, bit)
                    candidates.append((pm, bit, path))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._path_metric_penalty(cur_llr, bit)
                        candidates.append((pm, bit, path))

                for pm, bit, parent in candidates:
                    child = {
                        "pm": pm,
                        "L": parent["L"].copy(),
                        "B": parent["B"].copy(),
                        "u": parent["u"].copy(),
                    }
                    child["B"][l, n] = bit
                    child["u"][l] = bit

                    if l >= N / 2:
                        for s in range(n, n - _active_bit_level(l, n), -1):
                            block_size = int(2 ** s)
                            branch_size = block_size // 2
                            for j in range(l, -1, -block_size):
                                if j % block_size >= branch_size:
                                    child["B"][j - branch_size, s - 1] = (
                                        int(child["B"][j, s])
                                        ^ int(child["B"][j - branch_size, s])
                                    )
                                    child["B"][j, s - 1] = child["B"][j, s]

                    new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:self.list_size]

        # CRC 辅助路径选择
        if self.crc_length > 0:
            info_positions = sorted(set(range(N)) - self.frozen_set)
            crc_pass = [
                p for p in paths
                if crc_check(p["u"][info_positions], self.crc_length)
            ]
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
