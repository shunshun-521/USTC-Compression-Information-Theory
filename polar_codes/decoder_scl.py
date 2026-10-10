"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _frozen_mask, _reorder_channel_llr


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int) % 2
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int) % 2
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = _frozen_mask(frozen_bits)
        self.list_size = max(1, int(list_size))
        self.crc_length = int(crc_length)
        self.info_indices = np.flatnonzero(~self.frozen)

    def decode(self, llr_ch):
        llr = _reorder_channel_llr(llr_ch)
        self._metrics = [0.0]
        self._decisions = [np.zeros(self.N, dtype=int)]
        self._decode_node([llr], 0, self.N)

        paths = sorted(zip(self._metrics, self._decisions), key=lambda x: x[0])
        if self.crc_length > 0:
            for pm, dec in paths:
                payload = dec[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return dec, pm
        pm, u_hat = paths[0]
        return u_hat, pm

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr_vec in enumerate(llrs):
                self._metrics[path] += _penalty(float(llr_vec[0]), 0)
                self._decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self._metrics[path] + _penalty(float(llr_vec[0]), bit), path, bit)
            for path, llr_vec in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self._decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=int))
            parent_map.append(path)
        self._metrics = new_metrics
        self._decisions = new_decisions
        return betas, parent_map

    def _decode_node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(v[:half], v[half:]) for v in llrs]
        beta_upper, map_upper = self._decode_node(upper, base, half)

        lower = []
        for p in range(len(map_upper)):
            src = llrs[map_upper[p]]
            lower.append(g_operation(src[:half], src[half:], beta_upper[p]))
        beta_lower, map_lower = self._decode_node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
