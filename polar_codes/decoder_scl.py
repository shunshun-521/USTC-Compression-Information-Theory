"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np


_CRC8_POLY = 0x07
_CRC16_POLY = 0x1021  # CRC-16-CCITT


def _crc_bits_msb(data_bits, width, poly, init=0):
    """按 MSB-first 比特流计算 CRC 余数"""
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    crc = init & mask
    for b in data_bits:
        crc ^= int(b) << (width - 1)
        for _ in range(8):
            if crc & top:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    for mask in range(1 << r):
        crc_bits = np.array([(mask >> (r - 1 - i)) & 1 for i in range(r)], dtype=np.int8)
        full = np.concatenate([info_bits, crc_bits])
        if _crc_bits_msb(full, r, poly) == 0:
            return full
    raise RuntimeError("CRC encode failed")


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8).ravel()
    if crc_length == 8:
        poly, r = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, r = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    if len(bits) < r:
        return False
    return _crc_bits_msb(bits, r, poly) == 0


def _f(La, Lb):
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def _g(La, Lb, u):
    return Lb + (1.0 - 2.0 * u.astype(np.float64)) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL / SC（L=1）译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=np.int8)]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [_f(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [_g(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]
        _, _ = self._node([llr], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        best_u = None
        best_pm = None
        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_pm is None or pm < best_pm:
                        best_pm = pm
                        best_u = u_hat
        if best_u is None:
            best_pm, best_u = paths[0]
        return best_u.astype(np.int32), float(best_pm)
