"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def _sign_pm(x):
    """LLR 符号：0 视为正。"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


class _SCNode:
  def __init__(self, size, lane_id=-1, depth=0):
        self.size = size
        self.lane_id = lane_id
        self.depth = depth
        self.lambda_arr = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = False
        self.left = None
        self.right = None


def _build_polar_tree(depth):
    """构建与 aff3ct Binary_tree 相同 lane 编号的极化码树。"""
    lanes = [0] * (depth + 1)

    def create(cur_depth):
        if cur_depth > depth:
            return None
        lane = lanes[cur_depth]
        lanes[cur_depth] += 1
        if cur_depth == depth:
            return _SCNode(1, lane_id=lane, depth=cur_depth)
        node = _SCNode(2 ** (depth - cur_depth), lane_id=lane, depth=cur_depth)
        node.left = create(cur_depth + 1)
        node.right = create(cur_depth + 1)
        return node

    return create(0)


def _recursive_decode(node):
    """aff3ct Decoder_polar_SC_naive::recursive_decode 的 Python 实现。"""
    if node.left is not None:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_arr[i] = f_operation(
                node.lambda_arr[i], node.lambda_arr[half + i]
            )
        _recursive_decode(node.left)
        for i in range(half):
            node.right.lambda_arr[i] = g_operation(
                node.lambda_arr[i],
                node.lambda_arr[half + i],
                node.left.s[i],
            )
        _recursive_decode(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 1 if node.lambda_arr[0] < 0 else 0


def _assign_frozen(node, frozen_bits):
    if node.left is not None:
        _assign_frozen(node.left, frozen_bits)
        _assign_frozen(node.right, frozen_bits)
    else:
        node.is_frozen = bool(frozen_bits[node.lane_id])


def _collect_leaves(node, leaves):
    if node.left is None:
        leaves.append(node)
    else:
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)


def sc_decode(llr_ch, frozen_bits, apply_bit_reversal=True):
    """
    SC 译码主函数（树形实现，与 aff3ct naive SC 一致）。

    参数：
        llr_ch: 信道 LLR（与编码后码字逐位对应）
        frozen_bits: 1 表示冻结位
        apply_bit_reversal: 若编码器使用了 B_N，则对 LLR 做比特倒序
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    if apply_bit_reversal:
        from encoder import bit_reversal_permutation
        br = bit_reversal_permutation(N)
        llr_work = llr_ch[br].copy()
    else:
        llr_work = llr_ch.copy()

    root = _build_polar_tree(n)
    root.lambda_arr[:] = llr_work
    _assign_frozen(root, frozen_bits)
    _recursive_decode(root)

    u_hat = np.zeros(N, dtype=int)
    leaves = []
    _collect_leaves(root, leaves)
    for leaf in leaves:
        u_hat[leaf.lane_id] = leaf.s[0]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形实现别名）。"""
    return sc_decode(llr, frozen_bits, apply_bit_reversal=True)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供扩展使用）。"""
    n = int(math.log2(N))
    lambda_offset = [2 ** l for l in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while layer < n:
            if tmp & 1:
                break
            layers_llr.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while layer < n:
            if tmp & 1:
                layers_bit.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from encoder import polar_encode, polar_encode_butterfly, bit_reversal_permutation
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    np.random.seed(0)

    # 编码器校验：蝶形结果与 aff3ct 一致
    u = np.array([1, 0, 1, 1])
    x_br = polar_encode(u)
    x_bf = polar_encode_butterfly(u)
    print("butterfly:", x_bf, "with br:", x_br)

    for N in [4, 8, 16, 64, 256]:
        frozen = np.zeros(N, dtype=int)
        ok_br = 0
        ok_plain = 0
        for _ in range(50):
            u = np.random.randint(0, 2, N)
            x = polar_encode(u)
            llr = compute_llr(bpsk_modulate(x), 0.001)
            if np.array_equal(sc_decode(llr, frozen, True), u):
                ok_br += 1
            x2 = polar_encode_butterfly(u)
            llr2 = compute_llr(bpsk_modulate(x2), 0.001)
            if np.array_equal(sc_decode(llr2, frozen, False), u):
                ok_plain += 1
        print(f"N={N} decode ok: br={ok_br}/50 plain={ok_plain}/50")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen, True)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test N=64: {errors}/100 errors at 10dB")
