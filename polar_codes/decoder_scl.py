"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """计算 CRC 余数（输入含待校验数据，末尾已补零）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (fb * poly)
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
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _xor_combine(left, right):
    left = list(left)
    right = list(right)
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr_val, u_val):
        hard = 1 if llr_val < 0 else 0
        return 0.0 if u_val == hard else abs(llr_val)

    def _decode_paths(self, llr, depth, node, paths):
        """递归 SCL 路径扩展。"""
        if depth == self.n - 1:
            new_paths = []
            for pm, node_values, _ in paths:
                if node in self.frozen_set:
                    nv = node_values.copy()
                    nv[node] = 0
                    new_paths.append((pm, nv, [0]))
                else:
                    for bit in (0, 1):
                        penalty = abs(llr[0]) if (bit == 1) == (llr[0] >= 0) else 0.0
                        nv = node_values.copy()
                        nv[node] = bit
                        new_paths.append((pm + penalty, nv, [bit]))
            return new_paths

        half = len(llr) // 2
        l1 = llr[:half]
        l2 = llr[half:]
        left_llr = f_operation(l1, l2)

        left_paths = self._decode_paths(left_llr, depth + 1, 2 * node, paths)

        all_paths = []
        for pm, node_values, arr1 in left_paths:
            right_llr = g_operation(l1, l2, np.array(arr1))
            base_paths = [(pm, node_values, arr1)]
            right_paths = self._decode_paths(right_llr, depth + 1, 2 * node + 1, base_paths)
            for pm2, nv2, arr2 in right_paths:
                all_paths.append((pm2, nv2, _xor_combine(arr1, arr2)))

        all_paths.sort(key=lambda x: x[0])
        return all_paths[: self.list_size]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = np.asarray(llr_ch, dtype=np.float64)
        init_paths = [(0.0, [0] * self.N, [])]
        final_paths = self._decode_paths(llr, 0, 0, init_paths)

        candidates = []
        for pm, node_values, _ in final_paths:
            u_hat = np.array(node_values, dtype=int)
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            valid = [
                (pm, u) for pm, u in candidates
                if crc_check(u[self.info_indices], self.crc_length)
            ]
            pm, u_hat = min(valid, key=lambda x: x[0]) if valid else min(candidates, key=lambda x: x[0])
        else:
            pm, u_hat = min(candidates, key=lambda x: x[0])

        return u_hat, pm
