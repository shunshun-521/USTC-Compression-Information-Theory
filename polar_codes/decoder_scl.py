"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _active_bit_level, _active_llr_level, f_operation, g_operation


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径仅在写入时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.br = bit_reversal_permutation(N)

    @staticmethod
    def _path_metric_penalty(llr, u_bit):
        hard = 0 if llr >= 0.0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
                "owned_L": True,
                "owned_B": True,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = self.br[i]
            new_paths = []

            for path in paths:
                if not path["owned_L"]:
                    path["L"] = path["L"].copy()
                    path["owned_L"] = True
                if not path["owned_B"]:
                    path["B"] = path["B"].copy()
                    path["owned_B"] = True

                _update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    child = {
                        "L": path["L"],
                        "B": path["B"],
                        "pm": path["pm"] + self._path_metric_penalty(llr, 0),
                        "u_hat": path["u_hat"].copy(),
                        "owned_L": False,
                        "owned_B": False,
                    }
                    child["B"][l, n] = 0
                    child["u_hat"][l] = 0
                    if not child["owned_B"]:
                        child["B"] = child["B"].copy()
                        child["owned_B"] = True
                    _update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = {
                            "L": path["L"],
                            "B": path["B"],
                            "pm": path["pm"] + self._path_metric_penalty(llr, u_bit),
                            "u_hat": path["u_hat"].copy(),
                            "owned_L": False,
                            "owned_B": False,
                        }
                        child["B"][l, n] = u_bit
                        child["u_hat"][l] = u_bit
                        if not child["owned_B"]:
                            child["B"] = child["B"].copy()
                            child["owned_B"] = True
                        _update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass or paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
