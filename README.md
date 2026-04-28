# MRCR Eval / MRCR 评测

This repository runs OpenAI MRCR evaluations against OpenAI-compatible model APIs, writes per-row CSV results, optionally uses provider Batch APIs, and summarizes results by context-length thresholds.

本仓库用于运行 OpenAI MRCR 评测：调用兼容 OpenAI 接口的模型 API，生成逐行 CSV 结果；也支持部分供应商 Batch API；最后可以按上下文长度阈值汇总并画图。

## Setup / 环境准备

- Python 3.10+ is recommended.
- 推荐使用 Python 3.10+。
- Install dependencies / 安装依赖：
  ```bash
  pip install pandas huggingface_hub openai pyyaml matplotlib tiktoken pyarrow
  ```
- Configure models in `models.yaml`. Each model entry provides a `name`, `base_url`, `api_key`, and optional chat-completion parameters such as `temperature`, `max_tokens`, and `extra_body`.
- 在 `models.yaml` 中配置模型。每个模型需要提供 `name`、`base_url`、`api_key`，也可以配置 `temperature`、`max_tokens`、`extra_body` 等请求参数。
- `models.yaml` supports environment variable placeholders such as `${OPENROUTER_API_KEY}`, `${MOONSHOT_API_KEY}`, `${DASHSCOPE_API_KEY}`, `${DEEPSEEK_API_KEY}`, `${GLM_API_KEY}`, `${ARK_API_KEY}`, and `${MINIMAX_API_KEY}`.
- `models.yaml` 支持环境变量占位符，例如 `${OPENROUTER_API_KEY}`、`${MOONSHOT_API_KEY}`、`${DASHSCOPE_API_KEY}`、`${DEEPSEEK_API_KEY}`、`${GLM_API_KEY}`、`${ARK_API_KEY}`、`${MINIMAX_API_KEY}`。
- The dataset is downloaded at runtime from Hugging Face dataset `openai/mrcr`.
- 数据集会在运行时从 Hugging Face 的 `openai/mrcr` 下载。

## Online Evaluation / 在线评测 (`main.py`)

`main.py` loads the selected MRCR needle split, calls the configured model asynchronously, grades each response, and writes CSV rows with:

`main.py` 会加载指定的 MRCR needle 数据，异步调用模型，对每条回复打分，并写入以下 CSV 字段：

```text
grade, token_count, row
```

Rows whose estimated prompt length exceeds `--max-context-window` are skipped before sending requests. Token estimates use `tiktoken` encoding `o200k_base`; final `token_count` comes from API `usage.prompt_tokens`.

如果某行预估 prompt 长度超过 `--max-context-window`，会在请求前跳过。预估 token 使用 `tiktoken` 的 `o200k_base` 编码；最终写入的 `token_count` 来自 API 返回的 `usage.prompt_tokens`。

### CLI / 命令行参数

- `--needle NAME`: dataset subdirectory, default `8needle`. / 数据集子目录，默认 `8needle`。
- `--model-id NAME`: model name from `models.yaml`, default `kimi-k2.5`. / `models.yaml` 中的模型名，默认 `kimi-k2.5`。
- `--save-to PATH`: output CSV path. If the file exists, resume mode appends only missing rows. / 输出 CSV 路径；如果文件已存在，会进入续测模式，只追加缺失行。
- `--samples N`: number of calls per row, default `1`; grades are averaged. / 每行调用次数，默认 `1`；多次调用会取平均分。
- `--max-workers N`: concurrent worker count, default `20`. / 并发 worker 数，默认 `20`。
- `--max-context-window N`: max estimated prompt tokens. Default is `90%` of `256000`. / 最大预估 prompt token 数，默认是 `256000` 的 `90%`。

When `--save-to` is omitted, results are written to:

如果不传 `--save-to`，结果会写入：

```text
results/{needle}/{model}_{YYYYMMDD_HHMMSS}.csv
```

### Examples / 示例

```bash
python main.py --needle 8needle --model-id kimi-k2.5
```

```bash
python main.py --needle 8needle --model-id deepseek-v4-pro --save-to results/8needle/deepseek-v4-pro.csv
```

```bash
python main.py --needle 8needle --model-id kimi-k2.5 --samples 3 --max-workers 10
```

## Batch API Evaluation / Batch API 评测 (`batch_api/`)

`batch_api/moonshot/moonshot.py` and `batch_api/qwen/qwen.py` run the same MRCR grading flow through provider Batch APIs. They reuse dataset loading, model config, token estimation, grading, default CSV paths, and resume-row detection from `main.py`.

`batch_api/moonshot/moonshot.py` 和 `batch_api/qwen/qwen.py` 通过供应商 Batch API 运行同样的 MRCR 评测流程。它们复用 `main.py` 中的数据加载、模型配置、token 估算、打分、默认 CSV 路径和续测行检测逻辑。

Available scripts / 可用脚本：

- `batch_api/moonshot/moonshot.py`: default model `kimi-k2.6`, default artifacts dir `batch_api/moonshot/artifacts`. / 默认模型 `kimi-k2.6`，默认中间产物目录 `batch_api/moonshot/artifacts`。
- `batch_api/qwen/qwen.py`: default model `qwen3.6-flash`, default artifacts dir `batch_api/qwen/artifacts`. / 默认模型 `qwen3.6-flash`，默认中间产物目录 `batch_api/qwen/artifacts`。

Batch steps / Batch 步骤：

- `prepare`: build `batch_input.jsonl`, `row_payloads.json`, and `meta.json`. / 生成 `batch_input.jsonl`、`row_payloads.json` 和 `meta.json`。
- `upload`: upload the prepared JSONL file. / 上传准备好的 JSONL 文件。
- `create`: create the batch job. / 创建 Batch 任务。
- `wait`: poll until the batch reaches a terminal state. / 轮询直到 Batch 进入终态。
- `collect`: download output/error JSONL, grade responses, and append successful rows to CSV. / 下载输出和错误 JSONL，评分后把成功行追加到 CSV。
- `all`: run the whole pipeline in one command. / 一条命令运行完整流程。

`submit` is accepted as an alias for `upload`, and `poll` is accepted as an alias for `wait`.

`submit` 是 `upload` 的别名，`poll` 是 `wait` 的别名。

### Batch CLI / Batch 命令行参数

Common arguments / 通用参数：

- `--step {all,prepare,upload,create,wait,collect,submit,poll}`: default `all`. / 运行步骤，默认 `all`。
- `--needle NAME`: default `8needle`. / 数据集子目录，默认 `8needle`。
- `--model-id NAME`: model name from `models.yaml`. / `models.yaml` 中的模型名。
- `--save-to PATH`: result CSV path. Existing rows are skipped during `prepare`. / 结果 CSV 路径；`prepare` 阶段会跳过已有结果行。
- `--max-context-window N`: default `90%` of `256000`. / 最大预估 prompt token 数，默认 `256000` 的 `90%`。
- `--completion-window VALUE`: default `24h`. / Batch 完成窗口，默认 `24h`。
- `--poll-interval-seconds N`: default `10`. / 轮询间隔秒数，默认 `10`。
- `--artifacts-dir PATH`: where intermediate files are stored. / 中间产物保存目录。
- `--run-dir PATH`: required for `upload`, `create`, `wait`, and `collect`. / 执行 `upload`、`create`、`wait`、`collect` 时必填。
- `--batch-id ID`: optional override for `wait` and `collect`. / 可选 Batch ID 覆盖，用于 `wait` 和 `collect`。

### Batch Examples / Batch 示例

Run Moonshot end-to-end / 运行 Moonshot 完整流程：

```bash
python batch_api/moonshot/moonshot.py --step all --needle 8needle --model-id kimi-k2.6
```

Run Qwen end-to-end / 运行 Qwen 完整流程：

```bash
python batch_api/qwen/qwen.py --step all --needle 8needle --model-id qwen3.6-flash
```

Run step by step / 分步骤运行：

```bash
python batch_api/qwen/qwen.py --step prepare --needle 8needle --model-id qwen3.6-flash
python batch_api/qwen/qwen.py --step upload --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step create --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step wait --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step collect --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
```

Intermediate artifacts are stored under:

中间产物保存在：

```text
batch_api/{provider}/artifacts/{needle}/{model}/{timestamp}/
```

Important files include `batch_input.jsonl`, `batch_output.jsonl`, `batch_error.jsonl`, `row_payloads.json`, and `meta.json`.

重要文件包括 `batch_input.jsonl`、`batch_output.jsonl`、`batch_error.jsonl`、`row_payloads.json` 和 `meta.json`。

## Result Summary / 结果汇总 (`sorted_data.py`)

`sorted_data.py` reads CSV files from `results/{needle}`, filters them by filename keyword, computes cumulative average grades by token threshold, and writes a summary CSV plus a line chart.

`sorted_data.py` 会读取 `results/{needle}` 下的 CSV 文件，根据文件名关键字筛选模型，按 token 阈值计算累计平均分，并输出汇总 CSV 和折线图。

Current defaults / 当前默认值：

- `needle`: `8needle`
- thresholds / 阈值: `[8000, 16000, 32000, 64000, 128000, 256000, 512000, 900000]`
- output directory / 输出目录: `sorted_data/{needle}`

The keyword filter is required / 必须传入关键字过滤：

```bash
python sorted_data.py --keywords deepseek-v4-pro deepseek-v4-flash
```

Output files / 输出文件：

- `test_result_{timestamp}.csv`: rows by threshold, formatted as `model=avg|n=count`. / 按阈值输出汇总，格式为 `model=avg|n=count`。
- `test_result_{timestamp}.png`: line chart with token threshold on the x-axis and average grade on the y-axis. / 折线图，x 轴是 token 阈值，y 轴是平均分。

Threshold statistics are only calculated when the model has data reaching that threshold. For example, if a result file has data above `64000` but not above `128000`, the `128000` and higher threshold cells remain empty for that model.

只有当某个模型的数据达到对应阈值时，才会计算该阈值统计。例如某个结果文件有超过 `64000` 的数据，但没有超过 `128000` 的数据，那么该模型在 `128000` 及更高阈值上的统计会保持为空。

## Resume Behavior / 续测行为

- `main.py`: if `--save-to` points to an existing CSV, rows already present in the `row` column are skipped and remaining rows are appended.
- `main.py`：如果 `--save-to` 指向已存在的 CSV，会跳过 `row` 列中已有的行，只追加缺失行。
- Batch scripts: `prepare` reads the target CSV, excludes completed rows, and only submits missing rows.
- Batch 脚本：`prepare` 会读取目标 CSV，排除已完成行，只提交缺失行。
- If an existing CSV cannot be read, the scripts treat it as having no prior completed rows.
- 如果已有 CSV 无法读取，脚本会按没有历史结果处理。