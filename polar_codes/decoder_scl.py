"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _f_exact, _g_exact


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        mix = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = (reg << 1) & mask
        if mix:
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


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        decoder = _SCLCore(self.frozen_bits, self.list_size)
        paths = decoder.decode(llr_ch)

        if self.crc_length > 0:
            valid = []
            for metric, u_hat in paths:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((metric, u_hat))
            paths = valid if valid else paths

        best_metric, best_u = paths[0]
        return best_u.copy(), best_metric


class _SCLCore:
    def __init__(self, frozen_bits, list_size):
        self.frozen_bits = frozen_bits
        self.list_size = list_size
        self.block_length = len(frozen_bits)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]
        self._node([channel_llr], 0, self.block_length)
        paths = list(zip(self.metrics, self.decisions, strict=True))
        return sorted(paths, key=lambda item: item[0])

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda item: item[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=int))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [np.asarray(_f_exact(llr[:half], llr[half:]), dtype=np.float64) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        lower = [
            _g_exact(llrs[map_upper[p]][:half], llrs[map_upper[p]][half:], beta_upper[p])
            for p in range(len(map_upper))
        ]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(map_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
