"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import importlib.util
import math
import os
import numpy as np
from encoder import bit_reversal_permutation

_REF_FUNCTION_PATH = os.path.join(os.path.dirname(__file__), '_ref_function.py')
_spec = importlib.util.spec_from_file_location('polar_ref_function', _REF_FUNCTION_PATH)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return _ref.f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return _ref.g(La, Lb, u_hat)


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    return np.where(frozen_bits == 0)[0].tolist()


def _permute_llr_for_decode(llr_ch):
    """极化码编码含比特倒序，译码前对 LLR 做逆置换。"""
    N = len(llr_ch)
    inv = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode_nonrecursive(llr_ch, information_pos, frozen_bit=0):
    """非递归 SC 译码（位置遍历算法）。"""
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _ref.all_num(bit_matrix[n]) == 0:
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if _ref.all_num(up_bit) == 1:
            position = _ref.up(position)
        else:
            if _ref.all_num(right_bit) == 1:
                up_bit = _ref.get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit.copy()
            elif _ref.all_num(right_llr) == 1:
                if position[0] == position[2] - 1:
                    right_bit = _ref.get_right_bit(right_llr, information_pos, frozen_bit, position[1] + 1)
                    bit_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_bit
                else:
                    position = _ref.rightdown(position)
            elif _ref.all_num(left_bit) == 1:
                right_llr = _ref.get_right_llr(left_bit, up_llr)
                llr_matrix[position[0] + 1][position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr
            elif _ref.all_num(left_llr) == 0:
                left_llr = _ref.get_left_llr(up_llr)
                llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr
            elif position[0] == position[2] - 1:
                left_bit = _ref.get_left_bit(left_llr, information_pos, frozen_bit, position[1])
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_bit
            else:
                position = _ref.leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用高效分层实现作为参考）。"""
    return _sc_decode_layered(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量（接口兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while layer < n:
            if (tmp & 1) == 1:
                layers_llr.append(layer)
                tmp >>= 1
                layer += 1
            else:
                layers_llr.append(layer)
                break
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while layer < n:
            if (tmp & 1) == 0:
                layers_bit.append(layer)
                tmp >>= 1
                layer += 1
            else:
                break
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（首个 1 之前的 0 个数）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """比特回传起始层（首个 0 之前的 1 个数）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _sc_decode_layered(llr_ch, frozen_bits):
    """
    基于分层 LLR/比特数组的高效 SC 译码（mcba1n 风格，min-sum f/g）。
    信道 LLR 为自然顺序；内部按比特倒序相位译码。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    br = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if frozen_bits[br[l]]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    u_mcba = B[:, n].astype(int)
    return u_mcba[br]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（高效分层实现）。"""
    return _sc_decode_layered(llr_ch, frozen_bits)


def sc_decode_legacy(llr_ch, frozen_bits):
    """旧版参考实现（较慢，用于交叉验证）。"""
    info_pos = _frozen_to_info_pos(frozen_bits)
    llr_perm = _permute_llr_for_decode(llr_ch)
    return sc_decode_nonrecursive(llr_perm, info_pos, frozen_bit=0)


def verify_sc_decoders(N=64, K=32, num_trials=100, eb_n0_db=10.0):
    """验证 SC 译码器在高信噪比下无误。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_rec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"

    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC decoder verification passed.")
