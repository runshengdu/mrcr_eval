# MRCR Eval

This repository runs OpenAI MRCR evaluations against OpenAI-compatible model APIs, writes per-row CSV results, optionally uses provider Batch APIs, and summarizes results by context-length thresholds.

## Setup

- Python 3.10+ is recommended.
- Install dependencies:
  ```bash
  pip install pandas huggingface_hub openai pyyaml matplotlib tiktoken pyarrow
  ```
- Configure models in `models.yaml`. Each model entry provides a `name`, `base_url`, `api_key`, and optional chat-completion parameters such as `temperature`, `max_tokens`, and `extra_body`.
- `models.yaml` supports environment variable placeholders such as `${OPENROUTER_API_KEY}`, `${MOONSHOT_API_KEY}`, `${DASHSCOPE_API_KEY}`, `${DEEPSEEK_API_KEY}`, `${GLM_API_KEY}`, `${ARK_API_KEY}`, and `${MINIMAX_API_KEY}`.
- The dataset is downloaded at runtime from Hugging Face dataset `openai/mrcr`.

## Online Evaluation (`main.py`)

`main.py` loads the selected MRCR needle split, calls the configured model asynchronously, grades each response, and writes CSV rows with:

```text
grade, token_count, row
```

Rows whose estimated prompt length exceeds `--max-context-window` are skipped before sending requests. Token estimates use `tiktoken` encoding `o200k_base`; final `token_count` comes from API `usage.prompt_tokens`.

### CLI

- `--needle NAME`: dataset subdirectory, default `8needle`.
- `--model-id NAME`: model name from `models.yaml`, default `kimi-k2.5`.
- `--save-to PATH`: output CSV path. If the file exists, resume mode appends only missing rows.
- `--samples N`: number of calls per row, default `1`; grades are averaged.
- `--max-workers N`: concurrent worker count, default `20`.
- `--max-context-window N`: max estimated prompt tokens. Default is `90%` of `256000`.

When `--save-to` is omitted, results are written to:

```text
results/{needle}/{model}_{YYYYMMDD_HHMMSS}.csv
```

### Examples

```bash
python main.py --needle 8needle --model-id kimi-k2.5
```

```bash
python main.py --needle 8needle --model-id deepseek-v4-pro --save-to results/8needle/deepseek-v4-pro.csv
```

```bash
python main.py --needle 8needle --model-id kimi-k2.5 --samples 3 --max-workers 10
```

## Batch API Evaluation (`batch_api/`)

`batch_api/moonshot/moonshot.py` and `batch_api/qwen/qwen.py` run the same MRCR grading flow through provider Batch APIs. They reuse dataset loading, model config, token estimation, grading, default CSV paths, and resume-row detection from `main.py`.

Available scripts:

- `batch_api/moonshot/moonshot.py`: default model `kimi-k2.6`, default artifacts dir `batch_api/moonshot/artifacts`.
- `batch_api/qwen/qwen.py`: default model `qwen3.6-flash`, default artifacts dir `batch_api/qwen/artifacts`.

Batch steps:

- `prepare`: build `batch_input.jsonl`, `row_payloads.json`, and `meta.json`.
- `upload`: upload the prepared JSONL file.
- `create`: create the batch job.
- `wait`: poll until the batch reaches a terminal state.
- `collect`: download output/error JSONL, grade responses, and append successful rows to CSV.
- `all`: run the whole pipeline in one command.

`submit` is accepted as an alias for `upload`, and `poll` is accepted as an alias for `wait`.

### Batch CLI

Common arguments:

- `--step {all,prepare,upload,create,wait,collect,submit,poll}`: default `all`.
- `--needle NAME`: default `8needle`.
- `--model-id NAME`: model name from `models.yaml`.
- `--save-to PATH`: result CSV path. Existing rows are skipped during `prepare`.
- `--max-context-window N`: default `90%` of `256000`.
- `--completion-window VALUE`: default `24h`.
- `--poll-interval-seconds N`: default `10`.
- `--artifacts-dir PATH`: where intermediate files are stored.
- `--run-dir PATH`: required for `upload`, `create`, `wait`, and `collect`.
- `--batch-id ID`: optional override for `wait` and `collect`.

### Batch Examples

Run Moonshot end-to-end:

```bash
python batch_api/moonshot/moonshot.py --step all --needle 8needle --model-id kimi-k2.6
```

Run Qwen end-to-end:

```bash
python batch_api/qwen/qwen.py --step all --needle 8needle --model-id qwen3.6-flash
```

Run step by step:

```bash
python batch_api/qwen/qwen.py --step prepare --needle 8needle --model-id qwen3.6-flash
python batch_api/qwen/qwen.py --step upload --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step create --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step wait --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
python batch_api/qwen/qwen.py --step collect --run-dir batch_api/qwen/artifacts/8needle/qwen3.6-flash/20260423_120516
```

Intermediate artifacts are stored under:

```text
batch_api/{provider}/artifacts/{needle}/{model}/{timestamp}/
```

Important files include `batch_input.jsonl`, `batch_output.jsonl`, `batch_error.jsonl`, `row_payloads.json`, and `meta.json`.

## Result Summary (`sorted_data.py`)

`sorted_data.py` reads CSV files from `results/{needle}`, filters them by filename keyword, computes cumulative average grades by token threshold, and writes a summary CSV plus a line chart.

Current defaults:

- `needle`: `8needle`
- thresholds: `[8000, 16000, 32000, 64000, 128000, 256000, 512000, 900000]`
- output directory: `sorted_data/{needle}`

The keyword filter is required:

```bash
python sorted_data.py --keywords deepseek-v4-pro deepseek-v4-flash
```

Output files:

- `test_result_{timestamp}.csv`: rows by threshold, formatted as `model=avg|n=count`.
- `test_result_{timestamp}.png`: line chart with token threshold on the x-axis and average grade on the y-axis.

Threshold statistics are only calculated when the model has data reaching that threshold. For example, if a result file has data above `64000` but not above `128000`, the `128000` and higher threshold cells remain empty for that model.

## Resume Behavior

- `main.py`: if `--save-to` points to an existing CSV, rows already present in the `row` column are skipped and remaining rows are appended.
- Batch scripts: `prepare` reads the target CSV, excludes completed rows, and only submits missing rows.
- If an existing CSV cannot be read, the scripts treat it as having no prior completed rows.