# MRCR Test Code

本仓库包含两个主要脚本：

- `main.py`：运行/续测 MRCR 数据集的模型评测，按行异步调用模型接口并将结果写入 CSV。
- `sorted_data.py`：对评测结果进行阈值分桶统计并绘制折线图。

## 依赖与准备

- Python 3.10+（建议）。
- 依赖：`pandas`, `huggingface_hub`, `openai`（异步版 SDK）, `tiktoken`, `pyyaml`, `matplotlib` 等。可按需 `pip install pandas huggingface_hub openai tiktoken pyyaml matplotlib`.
- 需要在仓库根目录提供 `models.yaml`，包含模型的 `base_url` 与 `api_key`（可使用 `${ENV_VAR}` 占位并从环境变量读取）。`main.py` 默认模型名为 `glm-4.7`，如需改动请同步更新配置。
- 评测数据来自 HF 数据集 `openai/mrcr`，在运行时通过 `hf_hub_download` 自动下载。默认 needle 目录为 `2needle`。

## 运行评测（main.py）

`main.py` 会遍历 `needle` 对应的 parquet 数据（`2needle_0/1.parquet`），异步调用模型并将结果写入 CSV。写出的列：`grade`, `token_count`, `row`。

### CLI 参数

- `--save-to PATH`（可选）：指定结果 CSV 路径。
  - 若文件已存在，则进入“续测”模式，跳过已完成的 `row` 并追加剩余结果。
  - 若文件不存在，则新建并执行全量评测。
- `--samples N`（可选，默认 `SAMPLES` 常量=1）：每行调用次数，取平均分。

### 默认保存路径（未传 --save-to）

- 路径：`./results/{needle}/{model}_{timestamp}.csv`
- 其中 `{needle}` 和 `{model}` 来自脚本顶部常量（默认 `needle="2needle"`, `MODEL="glm-4.7"`），`{timestamp}` 为当前时间 `YYYYMMDD_HHMMSS`。

### 示例

- 全量评测并使用默认路径：
  ```bash
  python main.py
  ```
- 指定保存文件（不存在则新测，存在则续测）：
  ```bash
  python main.py --save-to results/2needle/glm-4.7_run.csv
  ```
- 调整采样次数：
  ```bash
  python main.py --samples 3 --save-to results/2needle/glm-4.7_run.csv
  ```

## 结果汇总与可视化（sorted_data.py）

`sorted_data.py` 读取 `results/{needle}` 下的 CSV，按预设关键词筛选模型文件，计算不同 token 阈值下的平均分并生成汇总表与折线图。

- 默认阈值列表：`[5000, 10000, 20000, 40000, 60000, 80000, 100000, 128000]`。
- 文件筛选关键词（文件名包含）：`["gpt-5.2","gemini-3","kimi-k2","minimax-m2.1","deepseek-v3.2-thinking","doubao","anthropic","glm-4.7"]`。
- 输出目录：`sorted_data/{needle}`，生成：
  - `test_result_{timestamp}.csv`：各阈值下各模型的平均分与样本数（格式 `model=avg|n=count`）。
  - `test_result_{timestamp}.png`：折线图（x 轴为最大 token 阈值，y 轴为平均分）。

### 运行

```bash
python sorted_data.py
```

运行前请确保 `results/{needle}` 目录下已有评测 CSV，且文件名满足关键词匹配。

## 重要常量（可按需调整）

- `needle`：默认 `"2needle"`，决定使用的数据子目录及输出路径。
- `MODEL`：默认 `"glm-4.7"`，影响默认输出文件名及模型配置读取。
- `CONCURRENCY`、`SAMPLES`、`MAX_RETRIES` 等参数位于 `main.py` 顶部。

## 续测机制说明

- 当 `--save-to` 指向已存在的 CSV 时，脚本读取已完成行的 `row` 列并跳过，仅对未完成行继续评测，结果追加到同一文件。
- 若读取已有结果出错，将回退为从头开始。