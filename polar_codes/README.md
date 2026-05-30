# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — 高斯近似（GA）+ SC 对齐探测
- `encoder.py` / `channel.py` — 编码与 AWGN-BPSK 信道
- `decoder_sc.py` — SC（SCD，主实现）与递归参考
- `decoder_scl.py` — SCL / CA-SCL（CRC-8/16）
- `decoder_bp.py` — BP（min-sum，早停）
- `simulation.py` / `utils.py` — 仿真与绘图
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

可通过环境变量加速调试：`POLAR_MAX_FRAMES=5000 POLAR_MIN_ERRORS=50`。

结果输出至 `results/`（CSV、PNG/PDF、frozen_sets.txt）。
