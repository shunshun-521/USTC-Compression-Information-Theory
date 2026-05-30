"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
    upper_llr,
    lower_llr,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = list(np.asarray(bits, dtype=int))
    nbytes = (len(bits) + 7) // 8
    out = bytearray()
    for i in range(nbytes):
        v = 0
        for j in range(8):
            if i * 8 + j < len(bits):
                v = (v << 1) | int(bits[i * 8 + j])
            else:
                v <<= 1
        out.append(v)
    return bytes(out)


def _crc8_bytes(data_bytes):
    crc = 0
    for b in data_bytes:
        crc ^= b
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ _CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bytes(data_bytes):
    crc = 0
    for b in data_bytes:
        crc ^= b << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ _CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（CRC-8: 0x07 / CRC-16: 0x8005）"""
    info_bits = np.asarray(info_bits, dtype=int)
    data = _bits_to_bytes(info_bits)
    if crc_length == 8:
        rem = _crc8_bytes(data)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc16_bytes(data)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    data = _bits_to_bytes(bits)
    if crc_length == 8:
        return _crc8_bytes(data) == 0
    if crc_length == 16:
        return _crc16_bytes(data) == 0
    raise ValueError("crc_length must be 8 or 16")


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_update(self, pm, llr, u):
        """路径度量更新：不一致时加 |LLR|"""
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        """
        SCL 译码主函数。
        返回 (u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        # 活跃路径：索引到共享 L/B 数组
        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan),
                "B": np.zeros((N, n + 1), dtype=int),
                "u": np.zeros(N, dtype=int),
                "active": True,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch.copy()

        for phi_nat in range(N):
            l = _bit_reversed(phi_nat, n)
            is_frozen = self.frozen_bits[l]
            candidates = []

            for p in paths:
                if not p["active"]:
                    continue
                L, B = p["L"], p["B"]

                for s in range(n - _active_llr_level(l, n), n):
                    block = 1 << (s + 1)
                    branch = block // 2
                    for j in range(l, N, block):
                        if j % block < branch:
                            L[j, s + 1] = upper_llr(L[j, s], L[j + branch, s])
                        else:
                            top_bit = B[j - branch, s + 1]
                            L[j, s + 1] = lower_llr(
                                L[j - branch, s], L[j, s], top_bit
                            )

                cur_llr = L[l, n]
                if is_frozen:
                    u = 0
                    pm = self._path_metric_update(p["pm"], cur_llr, u)
                    new_paths = [
                        {
                            "pm": pm,
                            "L": L.copy(),
                            "B": B.copy(),
                            "u": p["u"].copy(),
                            "active": True,
                        }
                    ]
                    new_paths[0]["u"][l] = u
                    new_paths[0]["B"][l, n] = u
                else:
                    new_paths = []
                    for u in (0, 1):
                        pm = self._path_metric_update(p["pm"], cur_llr, u)
                        np_ = {
                            "pm": pm,
                            "L": L.copy(),
                            "B": B.copy(),
                            "u": p["u"].copy(),
                            "active": True,
                        }
                        np_["u"][l] = u
                        np_["B"][l, n] = u
                        new_paths.append(np_)
                candidates.extend(new_paths)

            # 比特回传
            for p in candidates:
                if l < N // 2:
                    continue
                B = p["B"]
                for s in range(n, n - _active_bit_level(l, n), -1):
                    block = 1 << s
                    branch = block // 2
                    for j in range(l, -1, -block):
                        if j % block >= branch:
                            B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                            B[j, s - 1] = B[j, s]

            candidates.sort(key=lambda x: x["pm"])
            paths = candidates[: self.L_size]

        # 选择最优路径
        best = min(paths, key=lambda x: x["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x["pm"])

        return best["u"].astype(int), best["pm"]
