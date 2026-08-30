"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _update_llrs,
    _update_bits,
    _bit_reversed,
    f_operation,
    g_operation,
)


CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
        if msb ^ int(bit):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr_val, u_val):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, l, n)
                llr_val = L[l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_val, 0)
                    new_path = {
                        "pm": path["pm"] + pen,
                        "L": L.copy(),
                        "B": B.copy(),
                    }
                    new_path["B"][l, n] = 0
                    _update_bits(new_path["B"], l, n, N)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = {
                            "pm": path["pm"] + self._pm_penalty(llr_val, u_val),
                            "L": L.copy(),
                            "B": B.copy(),
                        }
                        new_path["B"][l, n] = u_val
                        _update_bits(new_path["B"], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                if crc_check(u_hat, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
