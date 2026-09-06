"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode_recursive, f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """CRC remainder (generator poly without leading 1)."""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for b in bits:
        reg = ((reg << 1) | int(b)) & mask
        if reg & top:
            reg = (reg ^ poly) & mask
    for _ in range(crc_length):
        reg = (reg << 1) & mask
        if reg & top:
            reg = (reg ^ poly) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _pm_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _compute_bit_llr(llr, u_prefix, phi):
    """基于已判决前缀计算第 phi 个比特的 LLR"""

    def decode_node(llr_node, bit_offset, length):
        if length == 1:
            return float(llr_node[0])
        half = length // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        if phi < bit_offset + half:
            return decode_node(llr_left, bit_offset, half)
        u_left = u_prefix[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        return decode_node(llr_right, bit_offset + half, half)

    return decode_node(np.asarray(llr, dtype=np.float64), 0, len(llr))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N

        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"bits": np.zeros(N, dtype=int), "pm": 0.0, "active": 0}]

        for phi in range(N):
            candidates = []
            for pidx, path in enumerate(paths):
                active = path["active"]
                prefix = path["bits"][:active].copy()
                full_prefix = path["bits"].copy()
                for i in range(active):
                    full_prefix[i] = path["bits"][i]
                llr = _compute_bit_llr(llr_ch, full_prefix, phi)

                if self.frozen_bits[phi]:
                    pm = path["pm"] + _pm_penalty(llr, 0)
                    candidates.append((pm, pidx, 0))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + _pm_penalty(llr, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit in candidates:
                src = paths[pidx]
                new_bits = src["bits"].copy()
                new_bits[phi] = bit
                new_paths.append({
                    "bits": new_bits,
                    "pm": pm,
                    "active": phi + 1,
                })
            paths = new_paths

        best = min(paths, key=lambda p: p["pm"])
        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path["bits"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path["pm"] < best_crc["pm"]:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best
        return chosen["bits"], chosen["pm"]
