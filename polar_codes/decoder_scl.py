"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, reorder_llr_for_decode


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, crc_length):
    if crc_length == 8:
        poly = _CRC8_POLY
        reg = 0
        for bit in bits:
            reg ^= (int(bit) << (crc_length - 1))
            for _ in range(8):
                if reg & (1 << (crc_length - 1)):
                    reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
                else:
                    reg = (reg << 1) & ((1 << crc_length) - 1)
        return reg
    if crc_length == 16:
        poly = _CRC16_POLY
        reg = 0
        for bit in bits:
            reg ^= (int(bit) << 15)
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        return reg
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length <= 0:
        return True
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """SCL 译码器（逐比特列表扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N)) + 1
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _metric_penalty(self, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数"""
        llr = reorder_llr_for_decode(llr_ch)
        N = self.N
        frozen = self.frozen_bits

        paths = [{"pm": 0.0, "u": np.zeros(N, dtype=int)}]

        for phi in range(N):
            candidates = []
            for path in paths:
                # 使用当前路径的部分译码结果计算 LLR
                llr_phi = self._compute_phi_llr(llr, phi, path["u"], frozen)
                if frozen[phi]:
                    pm = path["pm"] + self._metric_penalty(llr_phi, 0)
                    u_new = path["u"].copy()
                    u_new[phi] = 0
                    candidates.append({"pm": pm, "u": u_new})
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._metric_penalty(llr_phi, bit)
                        u_new = path["u"].copy()
                        u_new[phi] = bit
                        candidates.append({"pm": pm, "u": u_new})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"], best["pm"]

    def _compute_phi_llr(self, llr, phi, u_hat, frozen):
        """计算第 phi 个比特的 LLR（与递归 SC 一致，使用 u_hat[0:phi] 作为已知前缀）"""
        n = self.n
        frozen_set = set(np.where(frozen)[0])
        llr_result = [0.0]

        def _xor(left, right):
            return [(left[i] + right[i]) % 2 for i in range(len(left))] + list(right)

        def decode_node(y, depth, node):
            if depth == n - 1:
                if node == phi:
                    llr_result[0] = float(y[0])
                if node in frozen_set:
                    bit = 0
                elif node < phi:
                    bit = int(u_hat[node])
                elif node == phi:
                    bit = 0 if y[0] >= 0 else 1
                else:
                    bit = 0 if y[0] >= 0 else 1
                return [bit]

            half = len(y) // 2
            l1, l2 = np.asarray(y[:half]), np.asarray(y[half:])
            left_node = 2 * node
            arr1 = decode_node(f_operation(l1, l2).tolist(), depth + 1, left_node)
            arr2 = decode_node(
                g_operation(l1, l2, arr1).tolist(), depth + 1, left_node + 1
            )
            return _xor(arr1, arr2)

        decode_node(llr.tolist(), 0, 0)
        return llr_result[0]
