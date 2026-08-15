"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _prepare_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)

    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr_val, bit):
    """路径度量惩罚。"""
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit == hard else abs(llr_val)


class SCLDecoder:
    """
    SCL 译码器（树形递归 + 路径度量）。
    list_size=1 时等价于 SC 译码。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _leaf(self, llrs, index, metrics, decisions):
        """单叶节点译码。"""
        if self.frozen_bits[index]:
            new_metrics = metrics.copy()
            new_decisions = [d.copy() for d in decisions]
            for path, llr in enumerate(llrs):
                new_metrics[path] += _pm_penalty(float(llr[0]), 0)
                new_decisions[path][index] = 0
            betas = [np.array([0], dtype=int) for _ in llrs]
            return betas, list(range(len(llrs))), new_metrics, new_decisions

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit))

        candidates.sort(key=lambda item: item[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=int))
            parent_map.append(path)

        return betas, parent_map, new_metrics, new_decisions

    def _node(self, llrs, metrics, decisions, base, length):
        """递归译码子树。"""
        if length == 1:
            return self._leaf(llrs, base, metrics, decisions)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_up, map_up, metrics, decisions = self._node(
            upper, metrics, decisions, base, half
        )

        aligned_a = [llrs[map_up[p]][:half] for p in range(len(map_up))]
        aligned_b = [llrs[map_up[p]][half:] for p in range(len(map_up))]
        lower = [
            g_operation(aligned_a[p], aligned_b[p], beta_up[p])
            for p in range(len(map_up))
        ]
        beta_low, map_low, metrics, decisions = self._node(
            lower, metrics, decisions, base + half, half
        )

        beta_up = [beta_up[map_low[p]] for p in range(len(map_low))]
        betas = [
            np.concatenate([np.bitwise_xor(beta_up[p], beta_low[p]), beta_low[p]])
            for p in range(len(beta_low))
        ]
        parent_map = [map_up[map_low[p]] for p in range(len(map_low))]
        return betas, parent_map, metrics, decisions

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = _prepare_llr(llr_ch, self.N)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=int)]
        llrs = [np.asarray(llr_ch, dtype=np.float64)]

        _, _, metrics, decisions = self._node(llrs, metrics, decisions, 0, self.N)

        paths = sorted(zip(metrics, decisions, strict=True), key=lambda item: item[0])

        if self.crc_length > 0:
            valid = [(pm, dec) for pm, dec in paths
                     if crc_check(dec[self.info_indices], self.crc_length)]
            if valid:
                best_pm, best_dec = min(valid, key=lambda item: item[0])
            else:
                best_pm, best_dec = paths[0]
        else:
            best_pm, best_dec = paths[0]

        return best_dec.copy(), best_pm
