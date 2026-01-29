# MRCR Test Code

This repository contains two main scripts:

- `main.py`: Run/resume MRCR evaluations, asynchronously call the model API per row, and write results to CSV.
- `sorted_data.py`: Bucket results by token thresholds and plot line charts.

## Dependencies & Setup

- Python 3.10+ (recommended)
- Dependencies: `pandas`, `huggingface_hub`, `openai` (async SDK), `pyyaml`, `matplotlib`
  - Install as needed: `pip install pandas huggingface_hub openai pyyaml matplotlib`
- Provide a `models.yaml` in the repo root with `base_url` and `api_key` (supports `${ENV_VAR}` placeholders)
  - Default model is `kimi-k2.5`, overridable via `--model-id`
- The dataset comes from HF `openai/mrcr` and is downloaded at runtime via `hf_hub_download`
- Default needle directory: `2needle`

## Running Evaluations (`main.py`)

`main.py` iterates over the parquet files for the selected needle (`2needle_0/1.parquet`), calls the model asynchronously, and writes results to CSV. Columns:
`grade`, `token_count`, `row`, `prompt_char_count`, `real_prompt_tokens`, `updated_token_ratio`.

### CLI Arguments

- `--save-to PATH` (optional): output CSV path
  - If the file exists, resume mode is used and completed rows are skipped
  - If not, a full run is started
- `--samples N` (optional, default `SAMPLES`=1): number of calls per row (averaged)
- `--model-id NAME` (optional, default `kimi-k2.5`): model name (key in `models.yaml`)

### Default Output Path (when `--save-to` is omitted)

- Path: `./results/{needle}/{model}/{timestamp}.csv`
- `{needle}` defaults to `2needle`, `{model}` is `--model-id` (default `kimi-k2.5`), `{timestamp}` is `YYYYMMDD_HHMMSS`

### Examples

- Full run with default path:
  ```bash
  python main.py --model-id kimi-k2.5
  ```
- Specify output file (new or resume):
  ```bash
  python main.py --model-id kimi-k2.5 --save-to results/2needle/kimi-k2.5/run.csv
  ```
- Adjust samples:
  ```bash
  python main.py --samples 3 --model-id kimi-k2.5 --save-to results/2needle/kimi-k2.5/run.csv
  ```

## Result Summary & Visualization (`sorted_data.py`)

`sorted_data.py` reads CSVs under `results/{needle}`, filters by keywords, computes average grades by token thresholds, and generates a summary table and a line chart.

- Default thresholds: `[5000, 10000, 20000, 40000, 60000, 80000, 100000, 128000]`
- Filename keywords: `"gpt-5.2","gemini-3","kimi-k2","minimax-m2.1","deepseek-v3.2-thinking","doubao","anthropic","glm-4.7"`
- Output directory: `sorted_data/{needle}`
  - `test_result_{timestamp}.csv`: per-threshold averages and sample counts (`model=avg|n=count`)
  - `test_result_{timestamp}.png`: line chart (x=token threshold, y=average grade)

### Run

```bash
python sorted_data.py
```

Ensure `results/{needle}` contains CSVs that match the keyword filter before running.

## Key Constants (Adjustable)

- `needle`: default `"2needle"`, selects dataset subdir and output path
- `DEFAULT_MODEL_ID`/`MODEL`: default `"kimi-k2.5"`, controls model config and default output path
- `CONCURRENCY`, `SAMPLES`, `MAX_RETRIES`, etc. are defined at the top of `main.py`

## Resume Behavior

- If `--save-to` points to an existing CSV, completed rows are skipped and remaining results are appended
- Resume mode does **not** warm up; it loads the latest `updated_token_ratio` (or `real_prompt_tokens / prompt_char_count`) from the CSV and continues updating the ratio during resume
- If reading the existing CSV fails, the script falls back to a full run