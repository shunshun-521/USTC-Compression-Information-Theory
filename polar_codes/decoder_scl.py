"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(degree):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << degree) - 1)
            else:
                reg = (reg << 1) & ((1 << degree) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, degree = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, degree)
    crc_bits = np.array([(rem >> i) & 1 for i in range(degree - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly, degree = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, degree = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, degree) == 0


def _path_metric_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        return pm + abs(llr)
    return pm


def _partial_u_up(u_segment):
    """
    由已判决的源比特段计算 SC g 节点所需的 u_up（与 sc_decode_recursive 一致）。
    u_segment 长度为 2^layer。
    """
    u_segment = np.asarray(u_segment, dtype=int)
    n = len(u_segment)
    if n == 1:
        return u_segment.copy()
    half = n // 2
    u1_up = _partial_u_up(u_segment[:half])
    u2_up = _partial_u_up(u_segment[half:])
    u1_xor = np.bitwise_xor(u1_up, u2_up)
    return np.concatenate([u1_xor, u2_up])


def _llr_at_phi(u_prefix, phi, llr_ch, frozen_bits, n):
    """递归计算第 phi 位的 LLR（已知 u_prefix[0:phi]）。"""

    def rec(node_llr, layer, offset, target_phi, u_pref):
        if layer == 0:
            return float(node_llr[0])
        half = 1 << (layer - 1)
        llr_u = f_operation(node_llr[:half], node_llr[half:])
        if target_phi < offset + half:
            return rec(llr_u, layer - 1, offset, target_phi, u_pref)
        u_left = u_pref[offset : offset + half]
        u_left_up = _partial_u_up(u_left)
        llr_d = g_operation(node_llr[:half], node_llr[half:], u_left_up)
        return rec(llr_d, layer - 1, offset + half, target_phi, u_pref)

    return rec(llr_ch.copy(), n, 0, phi, u_prefix)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            from decoder_sc import sc_decode_recursive

            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr_phi = _llr_at_phi(path["u"], phi, llr_ch, self.frozen_bits, self.n)
                if self.frozen_bits[phi]:
                    u = 0
                    pm = _path_metric_update(path["pm"], llr_phi, u)
                    nu = path["u"].copy()
                    nu[phi] = u
                    new_paths.append({"pm": pm, "u": nu})
                else:
                    for u in (0, 1):
                        pm = _path_metric_update(path["pm"], llr_phi, u)
                        nu = path["u"].copy()
                        nu[phi] = u
                        new_paths.append({"pm": pm, "u": nu})
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])
            else:
                best = min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
