# 极化码编译码仿真（Python）

本目录实现 GA 构造、SC/SCL/BP 译码与蒙特卡洛仿真脚本，依赖见 `requirements.txt`。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python validate.py          # 单元校验
python construction.py      # 打印构造结果
python run_exp1.py          # 实验一（耗时长）
python run_exp2.py
python run_exp3.py
```

## 说明

- 编码采用 `G_N = B_N F^{⊗ n}`，信道 LLR 在译码前做比特倒序（`channel.decoder_llr`）。
- SC 提供递归参考实现与显式栈非递归实现；SCL 在 `L=1` 时与 SC 共用同一译码核心。
- 长码长（如 N≥256）下当前 SC 的 f 近似与列表译码路径更新仍可能偏离教科书曲线，建议以 `validate.py` 与短码 N=8 结果做回归，并按课程要求继续优化 SCL/BP 路径度量与因子图消息更新。
