"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _active_bit_level, _active_llr_level
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.array_equal(bits[-crc_length:], expected)


def _update_llrs(L, B, phase, n):
    start_layer = n - _active_llr_level(phase, n)
    for s in range(start_layer, n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(phase, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s],
                    L[j, s],
                    B[j - branch_size, s + 1],
                )
    return L[phase, n]


def _propagate_bits(B, phase, n, N):
    if phase < N // 2:
        return
    end_layer = n - _active_bit_level(phase, n)
    for s in range(n, end_layer, -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(phase, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（每条路径维护独立 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def _branch_metric(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        L0 = np.zeros((N, n + 1), dtype=np.float64)
        L0[:, 0] = llr_ch
        paths = [{"pm": 0.0, "L": L0, "B": np.zeros((N, n + 1), dtype=int)}]

        for i in range(N):
            phase = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                llr = _update_llrs(path["L"], path["B"], phase, n)

                if self.frozen_bits[phase]:
                    child = {
                        "pm": path["pm"] + self._branch_metric(llr, 0),
                        "L": path["L"],
                        "B": path["B"].copy(),
                    }
                    child["B"][phase, n] = 0
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._branch_metric(llr, bit),
                            "L": path["L"],
                            "B": path["B"].copy(),
                        }
                        child["B"][phase, n] = bit
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

            for path in paths:
                _propagate_bits(path["B"], phase, n, N)

        best_idx = 0
        if self.crc_length > 0:
            info_mask = self.frozen_bits == 0
            passed = []
            for idx, path in enumerate(paths):
                info_bits = path["B"][:, n][info_mask]
                if crc_check(info_bits, self.crc_length):
                    passed.append(idx)
            if passed:
                best_idx = min(passed, key=lambda i: paths[i]["pm"])

        u_hat = paths[best_idx]["B"][:, n].astype(int)
        return u_hat, paths[best_idx]["pm"]
