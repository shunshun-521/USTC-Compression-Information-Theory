"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import _f_scalar, _g_scalar


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in bits:
        feedback = (reg >> (crc_length - 1)) & 1
        reg = (reg << 1) & mask
        if int(b) ^ feedback:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_process(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_process(bits, crc_length, poly) == 0


def _path_metric(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u == u_hard:
        return pm
    return pm + abs(llr)


class _SCPathState:
    """单条路径的 SC 栈状态（与 decoder_sc.sc_decode 一致）"""

    def __init__(self, llr_ch, N):
        self.N = N
        self.u_hat = np.zeros(N, dtype=int)
        self.op_stack = [(llr_ch.tolist(), list(range(N)), 0, None)]
        self.partial_stack = []
        self.next_leaf = 0

    def clone(self):
        return copy.deepcopy(self)

    def leaf_llr_at(self, phi):
        """推进栈直至相位 phi 的叶 LLR 可用"""
        while True:
            if not self.op_stack:
                return 0.0
            llr_node, indices, phase, saved_enc_left = self.op_stack[-1]
            if len(indices) == 1:
                i = indices[0]
                if i < phi:
                    self.op_stack.pop()
                    self.partial_stack.append([self.u_hat[i]])
                    continue
                if i == phi:
                    return float(llr_node[0])
                return float(llr_node[0])
            half = len(indices) // 2
            if phase == 0:
                self.op_stack.pop()
                left_llr = [
                    _f_scalar(llr_node[i], llr_node[i + half]) for i in range(half)
                ]
                self.op_stack.append((llr_node, indices, 1, None))
                self.op_stack.append((left_llr, indices[:half], 0, None))
            elif phase == 1:
                self.op_stack.pop()
                enc_left = self.partial_stack.pop()
                right_llr = [
                    _g_scalar(llr_node[i], llr_node[i + half], enc_left[i])
                    for i in range(half)
                ]
                self.op_stack.append((llr_node, indices, 2, enc_left))
                self.op_stack.append((right_llr, indices[half:], 0, None))
            else:
                self.op_stack.pop()
                enc_right = self.partial_stack.pop()
                enc_left = saved_enc_left
                self.partial_stack.append(
                    [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right
                )

    def decide_bit(self, phi, u_bit, frozen):
        """在相位 phi 写入比特并完成叶节点压栈"""
        self.u_hat[phi] = 0 if frozen else u_bit
        if self.op_stack and len(self.op_stack[-1][1]) == 1:
            self.op_stack.pop()
        self.partial_stack.append([self.u_hat[phi]])
        while self.op_stack:
            llr_node, indices, phase, saved_enc_left = self.op_stack[-1]
            if len(indices) == 1:
                if indices[0] <= phi:
                    self.op_stack.pop()
                    continue
                break
            half = len(indices) // 2
            if phase == 0:
                self.op_stack.pop()
                left_llr = [
                    _f_scalar(llr_node[i], llr_node[i + half]) for i in range(half)
                ]
                self.op_stack.append((llr_node, indices, 1, None))
                self.op_stack.append((left_llr, indices[:half], 0, None))
            elif phase == 1:
                self.op_stack.pop()
                enc_left = self.partial_stack.pop()
                right_llr = [
                    _g_scalar(llr_node[i], llr_node[i + half], enc_left[i])
                    for i in range(half)
                ]
                self.op_stack.append((llr_node, indices, 2, enc_left))
                self.op_stack.append((right_llr, indices[half:], 0, None))
            else:
                self.op_stack.pop()
                enc_right = self.partial_stack.pop()
                enc_left = saved_enc_left
                self.partial_stack.append(
                    [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right
                )


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "state": _SCPathState(llr_ch, self.N)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                st = path["state"]
                llr_leaf = st.leaf_llr_at(phi)
                if self.frozen_bits[phi]:
                    ns = st.clone()
                    ns.decide_bit(phi, 0, True)
                    new_paths.append(
                        {
                            "pm": _path_metric(path["pm"], llr_leaf, 0),
                            "state": ns,
                        }
                    )
                else:
                    for u in (0, 1):
                        ns = st.clone()
                        ns.decide_bit(phi, u, False)
                        new_paths.append(
                            {
                                "pm": _path_metric(path["pm"], llr_leaf, u),
                                "state": ns,
                            }
                        )
            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        best = min(paths, key=lambda x: x["pm"])
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["state"].u_hat[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda x: x["pm"])
        return best["state"].u_hat.copy(), best["pm"]
