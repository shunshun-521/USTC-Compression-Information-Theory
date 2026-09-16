"""
AWGN 信道 + BPSK 调制/解调
与极化码 SC/SCL 译码器配套：0 -> -sqrt(Es), 1 -> +sqrt(Es)
"""
import numpy as np


def normalized_es(eb_n0_db, rate):
    """归一化符号能量 Es = 10^(Eb/N0/10) * R。"""
    return 10.0 ** (eb_n0_db / 10.0) * rate


def bpsk_modulate(x, eb_n0_db=None, rate=0.5):
    """
    BPSK 调制。
    若提供 eb_n0_db，则使用 Es 归一化；否则使用 0->+1, 1->-1。
    """
    x = np.asarray(x, dtype=np.float64)
    if eb_n0_db is None:
        return 1.0 - 2.0 * x
    es = normalized_es(eb_n0_db, rate)
    return 2.0 * (x - 0.5) * np.sqrt(es)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声。"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return np.asarray(s, dtype=np.float64) + noise


def compute_llr(y, sigma=None, eb_n0_db=None, rate=0.5):
    """
    计算 BPSK-AWGN 信道 LLR。
    优先使用 Es 归一化形式：LLR = -2 y sqrt(Es)。
    """
    y = np.asarray(y, dtype=np.float64)
    if eb_n0_db is not None:
        es = normalized_es(eb_n0_db, rate)
        return -2.0 * y * np.sqrt(es)
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """将 Eb/N0 (dB) 转换为噪声标准差（配套 Es 调制时 sigma = sqrt(1/2)）。"""
    _ = normalized_es(eb_n0_db, rate)
    return np.sqrt(0.5)


def hard_decision_llr(llr_ch):
    """对信道 LLR 做硬判决，返回码字估计（0/1）。"""
    return (llr_ch >= 0).astype(int)
