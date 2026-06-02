# 极化码编译码仿真

Python 实现（NumPy/SciPy），无第三方极化码库。

## 目录

- `construction.py` — GA 构造（自动校验与 SC 译码器兼容的信息位集合）
- `encoder.py` / `decoder_sc.py` / `decoder_scl.py` / `decoder_bp.py`
- `channel.py` / `simulation.py` / `utils.py`
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py`
- `results/` — 仿真输出（CSV、图像、冻结集）

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py          # 构造验证
python run_exp1.py              # SC 仿真
python run_exp2.py              # SCL / CA-SCL
python run_exp3.py              # BP 对比
```

快速冒烟（减少帧数、加宽 SNR 步进）：

```bash
POLAR_QUICK=1 POLAR_MAX_FRAMES=5000 POLAR_MIN_ERRORS=20 python run_exp1.py
```

## 说明

- LLR：`2y/σ²`，BPSK：`0→+1, 1→-1`
- SC 非递归实现采用分层 L/B 矩阵；输出经比特倒序映射回自然序
- 完整 BLER 曲线建议在较高 `Eb/N0`（约 6–10 dB，视码长而定）使用默认 `min_errors=100` 运行
