# 极化码编译码仿真（Python）

本目录实现极化码 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真（无第三方极化码库）。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python run_exp1.py   # SC
python run_exp2.py   # SCL / CA-SCL
python run_exp3.py   # BP 对比
```

快速冒烟（减少帧数）：

```bash
POLAR_QUICK=1 python run_exp1.py
```

## 模块

| 文件 | 说明 |
|------|------|
| `construction.py` | 高斯近似 GA 构造 |
| `encoder.py` | 蝶形编码 + 比特倒序 |
| `decoder_sc.py` | 递归/非递归 SC |
| `decoder_scl.py` | SCL 与 CRC 辅助 |
| `decoder_bp.py` | BP（min-sum + 早停） |
| `simulation.py` | 蒙特卡洛主循环 |
| `results/` | CSV 与曲线图 |

## 说明

- LLR 约定：`LLR = 2y/σ²`，正号倾向比特 0。
- 码率 `R=1/2` 时信息位取合成信道索引的高位半区，与 `B_N F^{⊗n}` 编码配套。
- 完整仿真可通过 `POLAR_MAX_FRAMES`、`POLAR_MIN_ERRORS` 调节。
