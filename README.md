# Compression × Information Theory（LLM 作为压缩器的实验复现）

本目录配套实验结果文档 `Information_Theory.pdf`，从**信息论的码长视角**出发，把压缩问题写成对序列概率的编码：

`L*(x1:n) = - sum_{i=1..n} log2 p(x_i | x_<i)`

并通过实验对比：

- **传统无损压缩**：`gzip / bzip2 / xz(LZMA) / zstd`
- **LLM 压缩（算术编码 + 语言模型概率）**：`GPTzip + GPT-2`、以及本仓库的 `Qwen2.5-0.5B` 版本

核心指标使用 **BPB（bits per byte）**：  
`BPB = |C(x)| * 8 / |x|`

其中 \(|C(x)|\) 是压缩后字节数，\(|x|\) 是原始字节数。BPB 越低表示压缩越好。

---

## 你会在这里找到什么

- **可复现实验脚本**：切片数据、跑传统压缩、跑 LLM 压缩、汇总 CSV、绘图
- **实验结果文件**：`results*.csv` 与 `fig*.png`
- **模型与数据目录**：`gpt2_models/`、`gpt2_medium_models/`、`qwen25_05b_models/`、`data/` 等

---

## 目录结构（重点文件）

- `Information_Theory.pdf`：报告正文（方法、推导、实验与结论）
- `enwik8`：数据集（Wikipedia 子集，100MB）
- `run_exp_1.py`：传统压缩基线实验（切片 + gzip/bzip2/xz/zstd）
- `run_llm.py`：批量跑 GPTzip（支持指定本地模型路径）
- `run_qwen.py` / `qwen_zip.py`：Qwen2.5-0.5B 作为概率模型的压缩/解压
- `plot_all.py`：合并多个结果 CSV 并生成对比图（fig5/fig6/fig7）
- `results.csv`：传统压缩结果
- `results_llm.csv`：GPTzip（通常对应 GPT-2 small）结果
- `results_GPTzip-medium.csv`：GPTzip + GPT-2 medium 结果
- `results_qwen.csv`：Qwen2.5-0.5B 结果
- `results_all.csv`：由 `plot_all.py` 合并生成

---

## 环境与依赖

报告中的实验环境大致为 Windows + PowerShell + Python 3.10 + PyTorch(CUDA) + transformers，并使用 Python 标准库与 `zstandard` 包进行传统压缩基线。

建议（最小）依赖：

- Python 3.10+
- `torch`（如需 GPU 推理）
- `transformers`
- `zstandard`
- `pandas`、`matplotlib`（生成汇总图表用）

---

## 快速开始

### 1) 生成 enwik8 切片 + 跑传统压缩基线

`run_exp_1.py` 会从 `enwik8` 读取前 1MB，并生成 4 个切片到 `data/`：
`1KB / 10KB / 100KB / 1MB`，然后对每个切片跑 4 种传统压缩器，输出 `results.csv`。

```bash
python run_exp_1.py
```

### 2) 跑 GPTzip（LLM 压缩）

`run_llm.py` 会对 `GPTzip/gptzip.py` 做一次“就地补丁”，把模型路径改为你传入的本地路径，并处理 Windows 文本换行与编码问题（以 `latin-1` 按 0–255 字节往返）。

示例（报告中的用法格式）：

```bash
python run_llm.py .\gpt2_models GPTzip 1KB 10KB 100KB 1MB
python run_llm.py .\gpt2_medium_models GPTzip-medium 1KB 10KB 100KB
```

输出：`results_<model_label>.csv`（例如 `results_GPTzip-medium.csv`）。

### 3) 跑 Qwen2.5-0.5B（LLM 压缩）

```bash
python run_qwen.py 1KB 10KB 100KB
```

输出：`results_qwen.csv`

### 4) 合并结果并绘图

```bash
python plot_all.py
```

输出：

- `results_all.csv`
- `fig5_length_bpb_all.png`：长度 vs BPB（传统 + LLM）
- `fig6_bpb_throughput_all.png`：BPB vs 吞吐（对数坐标）
- `fig7_llm_comparison.png`：三种 LLM 在不同切片上的 BPB 对比

---

## 方法概览（与信息论的对应关系）

- **编码长度与交叉熵**：当用模型 `q(x_i | x_<i)` 近似真实分布 `p*` 时，期望码长满足  
  `E[L] - H(X) = KL(p* || q) >= 0`，因此 `L >= H(X)`。  
  这也是“更好的概率模型 → 更短码长（更低 BPB）”的理论基础。

- **传统压缩器**：相当于不同的“隐式建模 + 编码”组合（如 LZ、BWT、熵编码等）。

- **LLM 压缩器**：显式用 LLM 给出条件概率，再用算术编码把概率变成比特流。

---

## 报告中的关键结论（摘要）

以下为 `Information_Theory.pdf` 的结论要点整理（具体数值、表格与图见 PDF 与本目录生成的 `results*.csv` / `fig*.png`）：

- **小文本（如 1KB）**：LLM 压缩在 BPB 上可能显著优于传统压缩（报告中 Qwen2.5-0.5B 在 1KB 取得极低 BPB 的示例）。
- **文本变长后**：LLM BPB 不一定单调变好，且吞吐量通常远低于传统压缩器（LLM 推理与 KV-cache 开销显著）。
- **模型大小/架构与 tokenizer 影响很大**：更强模型可能更低 BPB，但计算/显存成本也更高；不同 tokenizer（BPE 等）与字节往返策略会影响可压缩性与“无损”可逆性。
- **工程注意点**：Windows 下换行（`\n` vs `\r\n`）、按字节无损往返（`latin-1`）、以及 fp16/fp32 的确定性与数值误差，都可能影响解码正确性或实验可复现性。

---

## 常见问题（Troubleshooting）

- **GPTzip 在 Windows 下出现编码/换行差异**：本仓库的 `run_llm.py` 已对 `GPTzip/gptzip.py` 进行必要替换（`latin-1` + `newline=""`），避免 `\r\n` 与 UTF-8 编码带来的不一致。
- **LLM 解码失败或断言错误**：通常与 tokenizer、字节映射、或数值精度相关；建议先用最小切片（1KB）验证“往返无损”，再扩大长度。

---

## 引用与致谢

- `GPTzip/` 与相关代码：报告中引用了社区实现（如 `jxmorris12/gptzip`、`erika-n/GPTzip` 等），本仓库脚本在此基础上做了 Windows/本地模型路径适配与批量实验封装。
- 模型：`GPT-2`、`Qwen2.5-0.5B`（本地 `transformers` 推理）

