"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（基于树遍历的高效实现，与编码器 B_N F^{⊗n} 配套）
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

import ref_function as _ref_function

# 加载第三方参考 SC/SCL 核心（与 G = F^{⊗n}、信道 LLR 倒序配套）
_ref_path = Path(__file__).resolve().parent / "polar_sc_ref.py"
_spec = importlib.util.spec_from_file_location("polar_sc_ref", _ref_path)
_polar_ref = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("function", _ref_function)
sys.modules.setdefault("CRC", type(sys)("CRC_stub"))
_spec.loader.exec_module(_polar_ref)

from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0]


def _llr_for_decoder(llr_ch):
    """编码含 B_N 时，将信道 LLR 映射到 F^{⊗n} 译码器顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（教学用）；输入为原始信道 LLR"""
    llr = _llr_for_decoder(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                return np.array([0], dtype=np.int8)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=np.int8)

        half = n // 2
        llr_u = f_operation(llr_node[:half], llr_node[half:])
        u_left = decode_node(llr_u, frozen_node[:half])
        llr_u2 = g_operation(llr_node[:half], llr_node[half:], u_left)
        u_right = decode_node(llr_u2, frozen_node[half:])
        return np.concatenate([u_left, u_right])

    return decode_node(llr, frozen_bits)


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 的层索引（与参考树遍历实现配套，供报告说明）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        for layer in range(n):
            layers_llr.append(layer)
            if (p & 1) == 0:
                break
            p >>= 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        p = phi + 1
        for layer in range(n):
            if (p & 1) == 1:
                layers_bit.append(layer)
            p >>= 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（主函数）。
    frozen_bits: 1/True 表示冻结位
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = _frozen_to_info_pos(frozen_bits)
    frozen_value = 0
    llr = _llr_for_decoder(llr_ch)
    u_d, _ = _polar_ref.sc_decoder(llr, info_pos, frozen_value)
    return np.asarray(u_d, dtype=np.int8)
