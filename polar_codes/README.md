# 极化码编译码仿真（Python）

本目录实现极化码 GA 构造、蝶形编码、SC/SCL/BP 译码与蒙特卡洛仿真（无第三方极化码库）。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python construction.py          # 打印构造校验
python run_exp1.py              # SC 仿真
python run_exp2.py              # SCL / CA-SCL
python run_exp3.py              # BP 对比
```

快速模式（减少帧数）：

```bash
POLAR_QUICK=1 POLAR_MAX_FRAMES=5000 POLAR_MIN_ERRORS=30 python run_exp1.py
```

## 目录

- `construction.py` — GA 构造（信息位按比特倒序可靠性选取）
- `encoder.py` — 蝶形编码 + 比特倒序
- `decoder_sc.py` — SC（硬判决逆蝶形 + SSC 备选）
- `decoder_scl.py` — SCL（Chase 列表）+ CRC
- `decoder_bp.py` — BP（min-sum，含早停）
- `simulation.py` / `utils.py` — 仿真与绘图
- `results/` — CSV 与图像输出

## 编码校验

`polar_encode([1,0,1,1])` → `[1,0,1,1]`（与 $G_N=B_N F^{\otimes n}$ 一致）。
