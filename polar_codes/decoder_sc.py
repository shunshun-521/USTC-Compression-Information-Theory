"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import build_generator_matrix

_G_CACHE = {}


def _get_G(N):
    if N not in _G_CACHE:
        _G_CACHE[N] = build_generator_matrix(N)
    return _G_CACHE[N]


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _h(llr):
    return 0 if llr >= 0 else 1


def _sc_recursive_codeword(llr):
    """递归 SC：返回估计码字。"""
    lam = np.asarray(llr, dtype=np.float64)
    size = len(lam)
    if size == 1:
        return np.array([_h(lam[0])], dtype=np.int8)
    half = size // 2
    lam_l = f_operation(lam[:half], lam[half:])
    s_l = _sc_recursive_codeword(lam_l)
    lam_r = g_operation(lam[:half], lam[half:], s_l)
    s_r = _sc_recursive_codeword(lam_r)
    s = np.zeros(size, dtype=np.int8)
    for i in range(half):
        s[i] = s_l[i] ^ s_r[i]
        s[half + i] = s_r[i]
    return s


def _u_from_codeword(x_hat, frozen_bits):
    N = len(x_hat)
    frozen = np.asarray(frozen_bits, dtype=bool)
    G = _get_G(N)
    u_hat = (x_hat.astype(np.int64) @ G) % 2
    u_hat = u_hat.astype(np.int8)
    u_hat[frozen] = 0
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    return _u_from_codeword(_sc_recursive_codeword(llr), frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << layer)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while p % 2 == 1:
            llr_layers.append(layer)
            p //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi
        layer = 0
        while p % 2 == 1:
            bit_layers.append(layer)
            p //= 2
            layer += 1
        if phi % 2 == 0 and phi > 0:
            bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_nonrecursive_codeword(llr_ch):
    """非递归 SC：分治栈后序遍历，与递归版本等价。"""
    root = np.asarray(llr_ch, dtype=np.float64)
    tasks = [root]
    done = {}

    while tasks:
        node = tasks[-1]
        key = (node.shape[0], node.tobytes())
        if key in done:
            tasks.pop()
            continue
        if node.shape[0] == 1:
            done[key] = np.array([_h(node[0])], dtype=np.int8)
            tasks.pop()
            continue
        half = node.shape[0] // 2
        lam_l = f_operation(node[:half], node[half:])
        key_l = (lam_l.shape[0], lam_l.tobytes())
        if key_l not in done:
            tasks.append(lam_l)
            continue
        s_l = done[key_l]
        lam_r = g_operation(node[:half], node[half:], s_l)
        key_r = (lam_r.shape[0], lam_r.tobytes())
        if key_r not in done:
            tasks.append(lam_r)
            continue
        s_r = done[key_r]
        s = np.zeros(node.shape[0], dtype=np.int8)
        for i in range(half):
            s[i] = s_l[i] ^ s_r[i]
            s[half + i] = s_r[i]
        done[key] = s
        tasks.pop()

    root_key = (root.shape[0], root.tobytes())
    return done[root_key]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _u_from_codeword(_sc_nonrecursive_codeword(llr_ch), frozen_bits)
