"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _prepare_llr,
    _frozen_set_from_array,
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1

    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（与 SC 共享相同的 permuted 译码流程）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = _frozen_set_from_array(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = _prepare_llr(llr_ch)

        paths = [{
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
            "L": np.zeros((self.N, self.n + 1)),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                for s in range(self.n - active_llr_level(l, self.n), self.n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, self.N, block_size):
                        if j % block_size < branch_size:
                            path["L"][j, s + 1] = f_operation(
                                path["L"][j, s], path["L"][j + branch_size, s]
                            )
                        else:
                            path["L"][j, s + 1] = g_operation(
                                path["L"][j - branch_size, s],
                                path["L"][j, s],
                                path["B"][j - branch_size, s + 1],
                            )

                llr_val = path["L"][l, self.n]

                if l in self.frozen_set:
                    path["pm"] += self._pm_penalty(llr_val, 0)
                    path["B"][l, self.n] = 0
                    if l >= self.N // 2:
                        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                            block_size = 2 ** s
                            branch_size = block_size // 2
                            for j in range(l, -1, -block_size):
                                if j % block_size >= branch_size:
                                    path["B"][j - branch_size, s - 1] = (
                                        path["B"][j, s] ^ path["B"][j - branch_size, s]
                                    )
                                    path["B"][j, s - 1] = path["B"][j, s]
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = {
                            "pm": path["pm"] + self._pm_penalty(llr_val, u_bit),
                            "u_hat": path["u_hat"].copy(),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["B"][l, self.n] = u_bit
                        if l >= self.N // 2:
                            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                                block_size = 2 ** s
                                branch_size = block_size // 2
                                for j in range(l, -1, -block_size):
                                    if j % block_size >= branch_size:
                                        child["B"][j - branch_size, s - 1] = (
                                            child["B"][j, s] ^ child["B"][j - branch_size, s]
                                        )
                                        child["B"][j, s - 1] = child["B"][j, s]
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best_crc = None
        best_pm = None
        for path in paths:
            if self.crc_length > 0:
                info_bits = path["B"][:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path
            if best_pm is None or path["pm"] < best_pm["pm"]:
                best_pm = path

        chosen = best_crc if best_crc is not None else best_pm
        return chosen["B"][:, self.n].astype(int), chosen["pm"]
