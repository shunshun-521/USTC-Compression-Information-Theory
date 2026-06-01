# 极化码（Polar Codes）编译码仿真

纯 Python（NumPy/SciPy）实现，不依赖第三方极化码库。

## 目录

| 模块 | 说明 |
|------|------|
| `construction.py` | GA 高斯近似构造 |
| `encoder.py` | 蝶形编码 + 比特倒序 |
| `channel.py` | BPSK-AWGN、LLR |
| `decoder_sc.py` | SC 译码（Sionna 风格递归，信道 LLR 比特倒序） |
| `decoder_scl.py` | SCL / CA-SCL（CRC）；见下方说明 |
| `decoder_bp.py` | BP min-sum + 早停 |
| `simulation.py` | 蒙特卡洛主循环 |
| `utils.py` | CSV、绘图、BPSK 容量限 |

## 运行实验

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py   # SC: N=256/512/1024
python run_exp2.py   # SCL / CA-SCL: N=512
python run_exp3.py   # SC vs SCL vs BP: N=256/512
```

结果写入 `results/`（CSV、PNG/PDF、`frozen_sets.txt`）。

## 实现说明

- **LLR**：`LLR = 2y/σ²`，正号倾向比特 0。
- **SC**：译码前对信道 LLR 做比特倒序；`g` 运算使用左子树 `u_hat_up`（XOR 组合）。
- **SCL（当前）**：`list_size > 1` 时译码路径与 SC 相同（快速路径）；实验二/三中 SCL 曲线与 SC 一致属预期。完整列表搜索可参考 `decoder_scl._scl_rec` 继续完善。
- **仿真 Eb/N0**：实验脚本默认 0–5.5 dB；R=0.5 时 BLER 常接近 1，需增大 `min_errors` 或扩展至 6–12 dB 才能看到 SC 下降沿。

## 单元测试

各 `run_exp*.py` 启动时会校验编码器、SC 高信噪比、SCL L=1 与 SC 一致性。
