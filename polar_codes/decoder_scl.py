"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    mask = (1 << crc_length) - 1
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


def _path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _leaf(self, llrs, index, metrics, decisions):
        if self.frozen_bits[index]:
            new_metrics = []
            new_decisions = []
            betas = []
            parent_map = []
            for path, llr in enumerate(llrs):
                new_metrics.append(
                    metrics[path] + _path_metric_penalty(float(llr[0]), 0)
                )
                decision = decisions[path].copy()
                decision[index] = 0
                new_decisions.append(decision)
                betas.append(np.zeros(1, dtype=np.int8))
                parent_map.append(path)
            return new_metrics, new_decisions, betas, parent_map

        candidates = [
            (metrics[path] + _path_metric_penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)
        return new_metrics, new_decisions, betas, parent_map

    def _node(self, llrs, metrics, decisions, base, length):
        if length == 1:
            new_metrics, new_decisions, betas, parent_map = self._leaf(
                llrs, base, metrics, decisions
            )
            return betas, parent_map, new_metrics, new_decisions

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper, metrics, decisions = self._node(
            upper, metrics, decisions, base, half
        )

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower, metrics, decisions = self._node(
            lower, metrics, decisions, base + half, half
        )

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map, metrics, decisions

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]
        _, _, metrics, decisions = self._node([llr_ch], metrics, decisions, 0, self.N)

        paths = sorted(zip(metrics, decisions), key=lambda x: x[0])

        if self.crc_length > 0:
            valid = []
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((pm, u_hat))
            if valid:
                best_pm, best_u = min(valid, key=lambda x: x[0])
            else:
                best_pm, best_u = paths[0]
        else:
            best_pm, best_u = paths[0]

        return best_u.astype(int), best_pm
