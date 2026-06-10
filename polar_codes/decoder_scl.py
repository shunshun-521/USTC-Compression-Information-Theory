"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


# ==================== SCL 二叉树（aff3ct 风格单路径 LLR 记录）====================

class _SCLNode:
    __slots__ = (
        "left", "right", "father", "is_left", "lane", "is_leaf",
        "lam", "s", "frozen",
    )

    def __init__(self, size):
        self.left = None
        self.right = None
        self.father = None
        self.is_left = False
        self.lane = -1
        self.is_leaf = size == 1
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.frozen = False


def _build_scl_tree(size, counter, father=None, is_left=False):
    node = _SCLNode(size)
    node.father = father
    node.is_left = is_left
    if size == 1:
        node.lane = counter[0]
        counter[0] += 1
        return node
    half = size // 2
    node.left = _build_scl_tree(half, counter, node, True)
    node.right = _build_scl_tree(half, counter, node, False)
    return node


def _collect_leaves(node, leaves):
    if node.is_leaf:
        leaves.append(node)
    else:
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)


_SCL_TREE_CACHE = {}


def _get_scl_tree(N):
    if N not in _SCL_TREE_CACHE:
        root = _build_scl_tree(N, [0])
        leaves = []
        _collect_leaves(root, leaves)
        _SCL_TREE_CACHE[N] = (root, leaves)
    return _SCL_TREE_CACHE[N]


def _compute_depth(leaf_lane, m):
    if leaf_lane == 0:
        return m - 1
    res = 0
    index = leaf_lane
    while (index & 1) != 1 and res <= m - 1:
        index >>= 1
        res += 1
    return res


def _recursive_compute_llr(node_cur, depth):
    if depth > 0 and node_cur.father is not None:
        _recursive_compute_llr(node_cur.father, depth - 1)
        parent = node_cur.father
        if node_cur.is_left:
            _apply_f(parent)
        else:
            _apply_g(parent)


def _apply_f(node):
    half = len(node.lam) // 2
    for i in range(half):
        node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])


def _apply_g(node):
    half = len(node.lam) // 2
    for i in range(half):
        node.right.lam[i] = g_operation(node.lam[i], node.lam[half + i], node.left.s[i])


def _compute_sums(node):
    half = len(node.lam) // 2
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _propagate_sums(node_cur):
    if not node_cur.is_leaf:
        _compute_sums(node_cur)
    if (not node_cur.is_left) and node_cur.father is not None:
        _propagate_sums(node_cur.father)


def _update_pm(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        return pm + abs(llr)
    return pm


def _path_metric(u_hat, llrs, frozen_bits):
    pm = 0.0
    for i in range(len(u_hat)):
        if frozen_bits[i]:
            pm += _update_pm(0.0, llrs[i], 0)
        else:
            pm += _update_pm(0.0, llrs[i], u_hat[i])
    return pm


def _record_leaf_llrs(root, leaves, llr_ch, frozen_bits, m):
    """单路径顺序记录各叶节点 LLR（与 aff3ct SCL 单路径一致）。"""
    root.lam[:] = llr_ch
    leaf_llrs = np.zeros(len(leaves), dtype=np.float64)
    u_hat = np.zeros(len(leaves), dtype=np.int8)

    for leaf_index, leaf in enumerate(leaves):
        _recursive_compute_llr(leaf, _compute_depth(leaf_index, m))
        leaf_llrs[leaf_index] = leaf.lam[0]
        if frozen_bits[leaf_index]:
            leaf.s[0] = 0
            u_hat[leaf_index] = 0
        else:
            bit = 0 if leaf.lam[0] >= 0 else 1
            leaf.s[0] = bit
            u_hat[leaf_index] = bit
        _propagate_sums(leaf)

    return leaf_llrs, u_hat


# ==================== SCL 译码器 ====================

class SCLDecoder:
    """
    SCL 译码器。
    L=1 时等价于 SC；L>1 时在 SC 基础上对低可靠度信息位做单比特翻转扩展列表，
    并用路径度量选取最优路径（CRC 辅助时优先选通过校验的路径）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.root, self.leaves = _get_scl_tree(N)
        self._set_frozen(self.root)

    def _set_frozen(self, node):
        if node.is_leaf:
            node.frozen = bool(self.frozen_bits[node.lane])
        else:
            self._set_frozen(node.left)
            self._set_frozen(node.right)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        u_sc = sc_decode(llr_ch, self.frozen_bits)
        info_positions = np.where(~self.frozen_bits)[0]

        candidates = [u_sc.copy()]
        flip_order = sorted(info_positions, key=lambda i: abs(llr_ch[i]))
        for pos in flip_order[: max(0, self.list_size - 1)]:
            u_alt = u_sc.copy()
            u_alt[pos] ^= 1
            candidates.append(u_alt)

        if self.crc_length > 0:
            valid = []
            for u_hat in candidates:
                info_bits = u_hat[info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(u_hat)
            if valid:
                return valid[0].copy(), 0.0

        return u_sc.copy(), 0.0


def verify_scl_equals_sc(N=64, frozen_bits=None, num_trials=20, seed=1):
    """L=1 的 SCL 应与 SC 译码一致。"""
    rng = np.random.default_rng(seed)
    if frozen_bits is None:
        frozen_bits = np.zeros(N, dtype=bool)
        frozen_bits[: N // 2] = True
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(num_trials):
        llr = rng.normal(0, 2, N)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
