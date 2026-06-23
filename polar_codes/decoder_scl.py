"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _sc_decode_internal, sc_decode
from encoder import bit_reversal_permutation


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly_bits(crc_length):
    if crc_length == 8:
        return np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=int)
    if crc_length == 16:
        return np.array([1] + [int(b) for b in format(CRC16_POLY, "016b")], dtype=int)
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    for i in range(len(info_bits)):
        if msg[i]:
            msg[i : i + len(poly)] ^= poly
    return np.concatenate([info_bits, msg[len(info_bits) : len(info_bits) + crc_length]])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = _crc_poly_bits(crc_length)
    msg = bits.copy()
    for i in range(len(bits) - crc_length):
        if msg[i]:
            msg[i : i + len(poly)] ^= poly
    return not np.any(msg[-crc_length:])


class Path:
    """SCL 单条路径"""

    __slots__ = ("state", "pm")

    def __init__(self, n, N, llr_ch):
        self.state = {
            "n": n,
            "N": N,
            "intermediate_llr": _init_llr_layers(llr_ch),
            "intermediate_bits": [np.zeros(N, dtype=np.int8) for _ in range(n + 1)],
            "current_state": np.zeros(n, dtype=np.int8),
            "previous_state": np.ones(n, dtype=np.int8),
            "u_hat": np.zeros(N, dtype=int),
        }
        self.pm = 0.0


def _init_llr_layers(llr_ch):
    layers = [llr_ch.copy()]
    length = len(llr_ch) // 2
    while length > 0:
        layers.append(np.zeros(length, dtype=np.float64))
        length //= 2
    return layers


def _advance_path(path, position, frozen_br, n):
    from decoder_sc import (
        _compute_encoding_step,
        _compute_left_alpha,
        _compute_right_alpha,
        _position_state,
    )

    st = path.state
    st["current_state"] = _position_state(position, n)

    for i in range(1, n + 1):
        llr = st["intermediate_llr"][i - 1]
        if st["current_state"][i - 1] == st["previous_state"][i - 1]:
            continue
        if st["current_state"][i - 1] == 0:
            st["intermediate_llr"][i] = _compute_left_alpha(llr)
        else:
            end = position
            start = end - (1 << (n - i))
            left_bits = st["intermediate_bits"][i][start:end]
            st["intermediate_llr"][i] = _compute_right_alpha(llr, left_bits)

    llr = st["intermediate_llr"][-1][0]
    return llr


def _commit_bit(path, position, bit, frozen_br, n):
    from decoder_sc import _compute_encoding_step, _position_state

    st = path.state
    st["u_hat"][position] = bit
    st["intermediate_bits"][-1][position] = bit
    for i in range(n - 1, -1, -1):
        st["intermediate_bits"][i] = _compute_encoding_step(
            i, n, st["intermediate_bits"][i + 1], st["intermediate_bits"][i]
        )
    st["previous_state"][:n] = st["current_state"]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_br = self.frozen_bits[self.br]
        self.info_positions = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _llr_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if hard == bit else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [Path(self.n, self.N, llr_ch)]

        for position in range(self.N):
            llrs = [_advance_path(path, position, self.frozen_br, self.n) for path in paths]

            if self.frozen_br[position]:
                for path, llr in zip(paths, llrs):
                    path.pm += self._llr_penalty(llr, 0)
                    _commit_bit(path, position, 0, self.frozen_br, self.n)
                continue

            candidates = []
            for path_idx, (path, llr) in enumerate(zip(paths, llrs)):
                for bit in (0, 1):
                    candidates.append(
                        (path.pm + self._llr_penalty(llr, bit), path_idx, bit)
                    )

            candidates.sort(key=lambda x: x[0])
            keep = candidates[: self.list_size]

            new_paths = []
            for pm, path_idx, bit in keep:
                parent = paths[path_idx]
                child = Path(self.n, self.N, llr_ch)
                child.state = _clone_state(parent.state)
                child.pm = pm
                _commit_bit(child, position, bit, self.frozen_br, self.n)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_nat = path.state["u_hat"][self.br]
                info_bits = u_nat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.state["u_hat"][self.br].copy(), best.pm


def _clone_state(state):
    return {
        "n": state["n"],
        "N": state["N"],
        "intermediate_llr": [layer.copy() for layer in state["intermediate_llr"]],
        "intermediate_bits": [layer.copy() for layer in state["intermediate_bits"]],
        "current_state": state["current_state"].copy(),
        "previous_state": state["previous_state"].copy(),
        "u_hat": state["u_hat"].copy(),
    }
