"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np


def _f_exact(a, b):
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _g_exact(a, b, u):
    return b + (1.0 - 2.0 * u.astype(np.float64)) * a


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_run(bits, width, poly):
    reg = 0
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for b in np.asarray(bits, dtype=np.int8) % 2:
        reg ^= int(b) << (width - 1)
        for _ in range(width):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8) % 2
    if crc_length == 8:
        rem = _crc_run(info_bits, 8, _CRC8_POLY)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        rem = _crc_run(info_bits, 16, _CRC16_POLY)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits]).astype(np.int8)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8) % 2
    if crc_length == 8:
        rem = _crc_run(bits, 8, _CRC8_POLY)
        return rem == 0
    if crc_length == 16:
        rem = _crc_run(bits, 16, _CRC16_POLY)
        return rem == 0
    raise ValueError("crc_length must be 8 or 16")


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器（树递归 + 路径度量，L=1 时为 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_idx = np.flatnonzero(~self.frozen)

    def decode(self, llr_ch):
        from utils_llr import llr_to_decoder_order

        llr = llr_to_decoder_order(llr_ch)
        paths = self._scl_decode(llr)
        paths.sort(key=lambda p: p[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm
        u_hat, pm = paths[0][1], paths[0][0]
        return u_hat.astype(int), pm

    def _scl_decode(self, channel_llr):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]
        llr = np.asarray(channel_llr, dtype=np.float64)

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen[index]:
                for path, l in enumerate(llrs):
                    metrics[path] += _penalty(float(l[0]), 0)
                    decisions[path][index] = 0
                return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

            candidates = []
            for path, l in enumerate(llrs):
                for bit in (0, 1):
                    candidates.append((metrics[path] + _penalty(float(l[0]), bit), path, bit))
            candidates.sort(key=lambda x: x[0])
            kept = candidates[: self.list_size]

            new_metrics = []
            new_decisions = []
            betas = []
            parent_map = []
            for metric, path, bit in kept:
                new_metrics.append(metric)
                dec = decisions[path].copy()
                dec[index] = bit
                new_decisions.append(dec)
                betas.append(np.array([bit], dtype=np.int8))
                parent_map.append(path)
            metrics = new_metrics
            decisions = new_decisions
            return betas, parent_map

        def node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)
            half = length // 2
            upper = [_f_exact(l[:half], l[half:]) for l in llrs]
            beta_u, map_u = node(upper, base, half)
            a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
            b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
            lower = [_g_exact(a[p], b[p], beta_u[p]) for p in range(len(beta_u))]
            beta_l, map_l = node(lower, base + half, half)
            beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
            betas = [
                np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]]) for p in range(len(beta_l))
            ]
            parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
            return betas, parent_map

        _, _ = node([llr], 0, self.N)
        return [(metrics[i], decisions[i]) for i in range(len(metrics))]
