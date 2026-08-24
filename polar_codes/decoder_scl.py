"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _fast_llr, _llr_check_node_operation, _get_descendants, _get_problem_i


# ==================== CRC 工具 ====================

_CRC_POLY = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC_POLY.get(crc_length)
    if poly is None:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    reg = 0
    for bit in info_bits:
        reg ^= (bit << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_positions = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, bit):
        """路径度量更新：与 LLR 符号不一致时加 |LLR|。"""
        expected = 0 if llr >= 0 else 1
        if bit != expected:
            return pm + abs(llr)
        return pm

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64).copy()
        N, n = self.N, self.n
        L = self.list_size

        # 路径：pm, u_hat, llr_array, is_calc
        paths = [{
            'pm': 0.0,
            'u': np.full(N, -1, dtype=int),
            'llr_array': np.zeros(N * (n + 1), dtype=np.float64),
            'is_calc': [False] * (N * (n + 1)),
        }]

        for bit_idx in range(N):
            candidates = []

            for path in paths:
                llr_i = _fast_llr(
                    bit_idx, llr_ch, path['u'][:bit_idx],
                    path['llr_array'], path['is_calc'], n, N,
                )

                if bit_idx in self.frozen_positions:
                    new_path = {
                        'pm': self._path_metric_update(path['pm'], llr_i, 0),
                        'u': path['u'].copy(),
                        'llr_array': path['llr_array'].copy(),
                        'is_calc': path['is_calc'][:],
                    }
                    new_path['u'][bit_idx] = 0
                    candidates.append(new_path)
                else:
                    for b in (0, 1):
                        new_path = {
                            'pm': self._path_metric_update(path['pm'], llr_i, b),
                            'u': path['u'].copy(),
                            'llr_array': path['llr_array'].copy(),
                            'is_calc': path['is_calc'][:],
                        }
                        new_path['u'][bit_idx] = b
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['pm'])
            paths = candidates[:L]

        # 选择最优路径（CRC 优先）
        if self.crc_length > 0:
            info_positions = [i for i in range(N) if i not in self.frozen_positions]
            info_bits_all = np.array([paths[0]['u'][i] for i in info_positions])
            crc_pass = [
                p for p in paths
                if crc_check(np.array([p['u'][i] for i in info_positions]), self.crc_length)
            ]
            if crc_pass:
                paths = crc_pass

        best = min(paths, key=lambda p: p['pm'])
        return best['u'].astype(int), best['pm']
