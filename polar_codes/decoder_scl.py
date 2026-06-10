"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)

# ==================== CRC 工具 ====================

_CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length not in _CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if crc_length not in _CRC_POLYS:
        raise ValueError("crc_length must be 8 or 16")
    if len(bits) < crc_length:
        return False
    return _crc_remainder(bits, crc_length) == 0


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        """
        主译码函数。

        返回：
            u_hat: 长度 N 的估计源序列（最优路径）
            pm: 最优路径的度量值
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L_size = self.list_size

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int32),
            "u_hat": np.zeros(N, dtype=int),
            "active": True,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                if not path["active"]:
                    continue
                L, B = path["L"], path["B"]
                _update_llrs(L, B, l, n, N)
                llr = L[l, n]

                if self.frozen[l]:
                    candidates.append((path["pm"] + _path_metric_penalty(llr, 0),
                                       path, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (path["pm"] + _path_metric_penalty(llr, bit), path, bit)
                        )

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[:L_size]

            new_paths = []
            for pm, parent, bit in candidates:
                child = {
                    "pm": pm,
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "u_hat": parent["u_hat"].copy(),
                    "active": True,
                }
                child["u_hat"][l] = 0 if self.frozen[l] else bit
                child["B"][l, n] = 0 if self.frozen[l] else bit
                _update_bits(child["B"], l, n, N)
                new_paths.append(child)

            while len(new_paths) < L_size:
                new_paths.append({"active": False})
            paths = new_paths

        active_paths = [p for p in paths if p.get("active", False)]
        if not active_paths:
            return np.zeros(N, dtype=int), 0.0

        if self.crc_length > 0:
            valid = []
            for p in active_paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            pool = valid if valid else active_paths
        else:
            pool = active_paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
