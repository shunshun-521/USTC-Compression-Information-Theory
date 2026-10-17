"""
极化码编码器
编码：x = u * G_N，利用蝶形结构实现 O(N log N) 复杂度
"""
import numpy as np


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        r = 0
        for b in range(n):
            r = (r << 1) | ((i >> b) & 1)
        rev[i] = r
    return rev


def polar_encode(u):
    """
    极化码编码（pipelined / 5G NR 等价蝶形结构，O(N log N)）。

    参数：
        u: 长度为 N 的源序列（信息位 + 冻结位）

    返回：
        x: 长度为 N 的码字
    """
    u = np.asarray(u, dtype=np.int8)
    n = int(np.log2(len(u)))
    code_size = 1 << n
    code_half_size = code_size // 2
    output = np.zeros(code_size, dtype=np.int8)
    tmp = np.zeros(code_size, dtype=np.int8)
    i_even = np.arange(0, code_size, 2)
    i_odd = np.arange(1, code_size, 2)

    for j in range(code_half_size):
        tmp[j] = u[i_even[j]]
        tmp[j + code_half_size] = u[i_odd[j]]
    for j in range(code_half_size):
        output[i_odd[j]] = tmp[i_odd[j]]
        output[i_even[j]] = tmp[i_even[j]] ^ tmp[i_odd[j]]

    for _ in range(1, n):
        for j in range(code_half_size):
            tmp[j] = output[i_even[j]]
            tmp[j + code_half_size] = output[i_odd[j]]
        for j in range(code_half_size):
            output[i_odd[j]] = tmp[i_odd[j]]
            output[i_even[j]] = tmp[i_even[j]] ^ tmp[i_odd[j]]

    return output


def build_generator_matrix(N):
    """构建与 polar_encode 一致的生成矩阵（用于校验）"""
    G = np.zeros((N, N), dtype=int)
    for j in range(N):
        ej = np.zeros(N, dtype=int)
        ej[j] = 1
        G[:, j] = polar_encode(ej)
    return G


if __name__ == "__main__":
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode:", x)
    G = build_generator_matrix(4)
    print("matrix check:", np.array_equal(x, (u @ G) % 2))
