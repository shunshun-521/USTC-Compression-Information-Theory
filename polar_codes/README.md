# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码全流程：GA 构造、蝶形编码、SC / SCL / CA-SCL / BP 译码与蒙特卡洛 BLER 仿真。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python validate.py          # 单元校验
python run_exp1.py          # SC（N=256,512,1024）
python run_exp2.py          # SCL / CA-SCL（N=512）
python run_exp3.py          # BP 与对比（N=256,512）
```

输出目录：`results/`（CSV、曲线图、`frozen_sets.txt`）。

## 环境变量（加速或裁剪仿真）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `POLAR_MAX_FRAMES` | 100000 | 每信噪比点最大帧数 |
| `POLAR_MIN_ERRORS` | 100 | 每点最少错误帧数 |
| `POLAR_EB_MIN` / `MAX` / `STEP` | 0~5.5, 0.25 | Eb/N0 扫描范围 |
| `POLAR_SKIP_N1024` | 0 | 设为 `1` 时实验一跳过 N=1024 |

## 编码器约定

`N=4, u=[1,0,1,1]` 经蝶形与比特倒序后得 `x=[1,0,1,1]`（与生成矩阵一致）。部分教材手算示例为 `[0,0,1,1]`，对应不同的位序约定。

## 模块说明

- `construction.py` — 高斯近似（GA）构造
- `encoder.py` — O(N log N) 编码
- `channel.py` — BPSK-AWGN 与 LLR
- `decoder_sc.py` — 递归与非递归 SC
- `decoder_scl.py` — SCL、CRC-8/16（CA-SCL）
- `decoder_bp.py` — min-sum BP，早停
- `simulation.py` — 蒙特卡洛主循环
- `utils.py` — CSV、绘图、香农限
