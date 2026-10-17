"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _channel_llr_to_decoder,
    _active_llr_level,
    _active_bit_level,
    _upper_llr,
    _lower_llr,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后（CRC-8: 0x07, CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（路径复制实现，L=1 时等价于 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        fb = np.asarray(frozen_bits)
        self.frozen_set = set(np.where(fb.astype(bool) if fb.dtype == bool else fb != 0)[0])
        self.br = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, u):
        """与 LLR 符号不一致时加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _channel_llr_to_decoder(llr_ch)
        N, n, L = self.N, self.n, self.list_size

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = int(self.br[i])
            new_paths = []
            for path in paths:
                Lmat, Bmat = path["L"], path["B"]
                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            Lmat[j, s + 1] = _upper_llr(Lmat[j, s], Lmat[j + branch_size, s])
                        else:
                            top_bit = int(Bmat[j - branch_size, s + 1])
                            Lmat[j, s + 1] = _lower_llr(Lmat[j, s], Lmat[j - branch_size, s], top_bit)

                cur_llr = Lmat[l, n]
                if l in self.frozen_set:
                    u = 0
                    pm = path["pm"] + self._path_metric_penalty(cur_llr, u)
                    child = {
                        "pm": pm,
                        "L": Lmat.copy(),
                        "B": Bmat.copy(),
                        "u": path["u"].copy(),
                    }
                    child["u"][l] = u
                    child["B"][l, n] = u
                    self._update_bits(child["B"], l, n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = path["pm"] + self._path_metric_penalty(cur_llr, u)
                        child = {
                            "pm": pm,
                            "L": Lmat.copy(),
                            "B": Bmat.copy(),
                            "u": path["u"].copy(),
                        }
                        child["u"][l] = u
                        child["B"][l, n] = u
                        self._update_bits(child["B"], l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]

    def _update_bits(self, B, l, n):
        if l < self.N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
