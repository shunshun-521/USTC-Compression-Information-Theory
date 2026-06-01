# 极化码编译码仿真

纯 Python/NumPy/SciPy 实现的极化码（Polar Codes）构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 目录

- `construction.py` — GA 高斯近似构造
- `encoder.py` — 蝶形编码
- `decoder_sc.py` — SC 译码（Pfister 偶/奇递归）
- `decoder_scl.py` — SCL / CA-SCL
- `decoder_bp.py` — BP（min-sum + 早停）
- `simulation.py` — 蒙特卡洛主循环
- `run_exp1.py` / `run_exp2.py` / `run_exp3.py` — 实验脚本

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

快速冒烟（较少帧数、较短 Eb/N0 扫描）：

```bash
POLAR_QUICK=1 python run_exp1.py
POLAR_QUICK=1 python run_exp2.py
POLAR_QUICK=1 python run_exp3.py
```

环境变量：`POLAR_MAX_FRAMES`、`POLAR_MIN_ERRORS`。

## 单元测试

各实验脚本启动时会调用 `validate.py` 中的编码器、SC、CRC、L=1 SCL 一致性检查。
