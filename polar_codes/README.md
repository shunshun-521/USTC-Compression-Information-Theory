# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — 编码 `x = u @ G_N`
- `decoder_sc.py` — SC 译码（P/C 非递归主实现 + 递归参考 + 因子图矩阵遍历供 SCL）
- `decoder_scl.py` — SCL / CA-SCL
- `decoder_bp.py` — BP（min-sum，早停）
- `channel.py` — BPSK-AWGN
- `simulation.py` — 蒙特卡洛主循环
- `utils.py` — CSV、绘图、容量限
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本
- `results/` — 输出 CSV 与图像

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py          # 打印构造校验
python run_exp1.py              # SC 仿真
python run_exp2.py              # SCL / CA-SCL
python run_exp3.py              # BP 对比
```

快速冒烟（较少帧数、较粗 SNR 步进）：

```bash
POLAR_QUICK=1 python run_exp1.py
POLAR_QUICK=1 python run_exp2.py
POLAR_QUICK=1 python run_exp3.py
```

完整仿真请省略 `POLAR_QUICK`（默认 `max_frames=100000`, `min_errors=100`）。
