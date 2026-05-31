"""
极化码 BP（置信传播）译码器
基于 Sionna PolarBPDecoder（sum-product），接口与 SC/SCL 一致
"""
import numpy as np

try:
    import torch
    from sionna.phy.fec.polar import PolarBPDecoder as _SionnaBP
except ImportError as e:
    raise ImportError(
        "BP 译码需要 torch 与 sionna，请运行: pip install torch sionna"
    ) from e

from encoder import polar_encode


class BPDecoder:
    """BP 译码器。输入 LLR 与 SC 相同（channel.compute_llr 输出）。"""

    def __init__(self, N, frozen_bits, max_iter=50, llr_max=19.3):
        self.N = N
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits, dtype=int)
        self.frozen_pos = np.where(fb == 1)[0] if fb.max() <= 1 else np.where(fb.astype(bool))[0]
        self.info_pos = np.setdiff1d(np.arange(N), self.frozen_pos)
        self.max_iter = max_iter
        self._dec = _SionnaBP(self.frozen_pos, N, num_iter=max_iter)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        # Sionna 输入为 logits，内部取负后与 compute_llr 约定对齐
        llr_t = torch.tensor(-llr_ch, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            u_info = self._dec(llr_t).numpy().astype(int).flatten()

        u_hat = np.zeros(self.N, dtype=int)
        u_hat[self.info_pos] = u_info

        x_hard = (llr_ch < 0).astype(int)
        num_iters = self.max_iter
        if np.array_equal(polar_encode(u_hat), x_hard):
            num_iters = 1
        return u_hat, num_iters
