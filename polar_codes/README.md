# 极化码编译码仿真

基于 NumPy/SciPy 的极化码（Polar Codes）仿真框架，不依赖第三方极化码库。

## 目录结构

- `construction.py` — GA 高斯近似构造
- `encoder.py` — 蝶形 O(N log N) 编码
- `decoder_sc.py` — SC 译码（非递归 Permuted SC + 递归参考）
- `decoder_scl.py` — SCL / CA-SCL（CRC-8/16）
- `decoder_bp.py` — BP 译码（min-sum + 早停）
- `channel.py` — BPSK-AWGN 信道
- `simulation.py` — 蒙特卡洛仿真
- `utils.py` — CSV、绘图、容量限
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

可通过环境变量加速调试：

```bash
export POLAR_MAX_FRAMES=5000 POLAR_MIN_ERRORS=30
python run_exp1.py
```

## 单元测试

```bash
python test_modules.py
```
