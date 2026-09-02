"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 LLR 阵列）
"""
import numpy as np
from encoder import bit_reversal_permutation


def bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """从最高位起统计前导 0 个数 + 1"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """从最高位起统计前导 1 个数 + 1"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（LLR 域）"""
    return (1 - 2 * u_hat) * La + Lb


def permute_llr_for_decode(llr):
    """
    编码器输出 x = v[brp]，信道 LLR[i] 对应 x[i]=v[brp[i]]。
    译码树节点 i 需要 v[i] 的 LLR，即 llr_ch[inv_brp[i]]。
    """
    N = len(llr)
    brp = bit_reversal_permutation(N)
    inv_brp = np.argsort(brp)
    return np.asarray(llr, dtype=np.float64)[inv_brp]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码"""
    llr = permute_llr_for_decode(llr)
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """返回译码顺序（比特倒序）及每层活跃级数（供参考）"""
    n = int(np.log2(N))
    decode_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_levels = [active_llr_level(l, n) for l in decode_order]
    bit_levels = [active_bit_level(l, n) for l in decode_order]
    return decode_order, llr_levels, bit_levels


_SC_INDEX_CACHE = {}


class _SCDState:
    """分层 SC 译码内部状态"""

    def __init__(self, N):
        self.N = N
        self.n = int(np.log2(N))
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)


def _update_llrs(state, l):
    n = state.n
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, state.N, block_size):
            if j % block_size < branch_size:
                state.L[j, s + 1] = f_operation(state.L[j, s], state.L[j + branch_size, s])
            else:
                top_bit = int(state.B[j - branch_size, s + 1])
                state.L[j, s + 1] = g_operation(
                    state.L[j - branch_size, s], state.L[j, s], top_bit
                )


def _update_bits(state, l):
    if l < state.N / 2:
        return
    n = state.n
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                state.B[j - branch_size, s - 1] = int(state.B[j, s]) ^ int(
                    state.B[j - branch_size, s]
                )
                state.B[j, s - 1] = state.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits[i]=1 表示冻结位。
    """
    llr_ch = permute_llr_for_decode(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_set = set(np.where(frozen_bits == 1)[0])
    N = len(llr_ch)
    n = int(np.log2(N))

    state = _SCDState(N)
    state.L[:, 0] = llr_ch

    for phi in range(N):
        l = bit_reversed_index(phi, n)
        _update_llrs(state, l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        _update_bits(state, l)

    return state.B[:, n].astype(int)
