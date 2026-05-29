# 极化码编译码仿真

Python 实现的极化码（Polar Codes）构造、编码与译码仿真，不依赖第三方极化码库。

## 目录结构

- `construction.py` — GA 高斯近似构造
- `encoder.py` — 蝶形编码 + 比特倒序
- `decoder_sc.py` — SC 译码（非递归分层实现）
- `decoder_scl.py` — SCL / CA-SCL 译码
- `decoder_bp.py` — BP 译码（min-sum）
- `channel.py` — BPSK-AWGN 信道
- `simulation.py` — 蒙特卡洛仿真
- `utils.py` — 结果保存与绘图
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本
- `results/` — 输出 CSV 与图像

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

快速模式（较少帧数，用于 CI/调试）：

```bash
POLAR_FAST=1 python run_exp1.py
```

## 说明

- SC/SCL 译码器在比特倒序域与发送码字 LLR 对齐，高信噪比下与编码器一致。
- 完整蒙特卡洛仿真（`MAX_FRAMES=100000`）耗时较长，建议在本地或服务器上单独运行。
