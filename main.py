import argparse
from huggingface_hub import hf_hub_download
import pandas as pd
from openai import AsyncOpenAI
import json
from difflib import SequenceMatcher
import os
from datetime import datetime
from pathlib import Path
import asyncio
import csv
import re
from typing import Any, Awaitable, Callable
import yaml
import tiktoken

MODEL = ""
needle = ""
SAMPLES = 1
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 0

_enc = tiktoken.get_encoding("o200k_base")


def n_tokens(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if c is None:
            continue
        if isinstance(c, str):
            total += len(_enc.encode(c))
        else:
            total += len(_enc.encode(str(c)))
    return total


def default_max_context_window() -> int:
    return int(256000 * 0.9)


def _run_params(max_context_window: int | None, max_workers: int) -> tuple[int, int]:
    mcw = default_max_context_window() if max_context_window is None else int(max_context_window)
    return mcw, max(1, int(max_workers))


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

def safe_filename_component(s: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]+", "_", str(s))

def build_default_csv_path(model: str, selected_needle: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = safe_filename_component(model)
    safe_needle = safe_filename_component(selected_needle)
    result_dir = Path("results") / safe_needle
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / f"{safe_model}_{timestamp}.csv"


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, str):
        def _repl(m: re.Match) -> str:
            var_name = m.group(1)
            env_val = os.environ.get(var_name)
            if env_val is None:
                raise KeyError(
                    f"Environment variable '{var_name}' is required by models.yaml but is not set."
                )
            return env_val

        return _ENV_VAR_PATTERN.sub(_repl, value)
    if isinstance(value, list):
        return [_expand_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    return value


def load_model_config(model_name: str) -> dict[str, Any]:
    config_path = Path(__file__).with_name("models.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"models.yaml not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError("models.yaml must be a mapping at the top level")

    default_cfg = cfg.get("default") or {}
    models_cfg = cfg.get("models") or []

    if not isinstance(default_cfg, dict):
        raise ValueError("models.yaml: 'default' must be a mapping")
    if not isinstance(models_cfg, list):
        raise ValueError("models.yaml: 'models' must be a list")

    matched = None
    available: list[str] = []
    for m in models_cfg:
        if isinstance(m, dict) and m.get("name"):
            available.append(str(m.get("name")))
        if isinstance(m, dict) and m.get("name") == model_name:
            matched = m
            break

    if matched is None:
        available_sorted = sorted(set(available))
        raise ValueError(
            "Model name not found in models.yaml: "
            f"{model_name}. Available models: {available_sorted}"
        )

    merged: dict[str, Any] = dict(default_cfg)
    merged.update(matched)
    merged = _expand_env_vars(merged)

    base_url = merged.get("base_url")
    api_key = merged.get("api_key")
    if not base_url or not api_key:
        raise ValueError(
            f"models.yaml config for model '{model_name}' must provide non-empty 'base_url' and 'api_key'"
        )

    return merged


_CLIENT_CACHE: dict[tuple[str, str], AsyncOpenAI] = {}


def _client_from_config(cfg: dict[str, Any]) -> AsyncOpenAI:
    cache_key = (str(cfg["base_url"]), str(cfg["api_key"]))
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    new_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    _CLIENT_CACHE[cache_key] = new_client
    return new_client


def get_client_for_model(model_name: str) -> AsyncOpenAI:
    return _client_from_config(load_model_config(model_name))


_CHAT_COMPLETION_CONFIG_KEYS = {
    "temperature",
    "max_tokens",
    "extra_body",
}


def build_chat_completion_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        k: model_cfg[k]
        for k in _CHAT_COMPLETION_CONFIG_KEYS
        if k in model_cfg and model_cfg[k] is not None
    }

MODEL_CONFIG: dict[str, Any] | None = None
client: AsyncOpenAI | None = None

def init_model(model_id: str) -> None:
    global MODEL, MODEL_CONFIG, client
    MODEL = str(model_id)
    MODEL_CONFIG = load_model_config(MODEL)
    client = _client_from_config(MODEL_CONFIG)

def load_dataset(selected_needle: str) -> pd.DataFrame:
    selected_needle = str(selected_needle)
    parts = []
    for suffix in ("_0", "_1"):
        parts.append(
            pd.read_parquet(
                hf_hub_download(
                    repo_id="openai/mrcr",
                    filename=f"{selected_needle}/{selected_needle}{suffix}.parquet",
                    repo_type="dataset",
                )
            )
        )
    return pd.concat(parts)

dataset: pd.DataFrame | None = None

def grade(response, answer, random_string_to_prepend) -> float:
    """
    Compare response and answer with strict prefix enforcement.
    """
    # Normalize leading BOM/zero-width/whitespace before prefix check
    def normalize_leading(s: str) -> str:
        return re.sub(r"^[\ufeff\u200b\u200c\u200d\s]+", "", s.replace("\r\n", "\n"))

    response_norm = normalize_leading(response)
    if not response_norm.startswith(random_string_to_prepend):
        return 0.0

    response_body = response_norm.removeprefix(random_string_to_prepend).strip()
    answer_body = normalize_leading(answer).removeprefix(random_string_to_prepend).strip()
    return float(SequenceMatcher(None, response_body, answer_body).ratio())

def _read_existing_csv_header(filename: Path) -> list[str] | None:
    try:
        if not filename.exists() or filename.stat().st_size == 0:
            return None
        with open(filename, "r", encoding="utf-8", newline="") as f:
            first_line = f.readline().strip()
        if not first_line:
            return None
        parts = [p.strip() for p in first_line.split(",") if p.strip()]
        return parts or None
    except Exception:
        return None

async def csv_writer(queue: asyncio.Queue, filename: Path):
    existing = _read_existing_csv_header(filename)
    if existing:
        header_written = True
        fieldnames = existing
    else:
        header_written = False
        fieldnames = ["grade", "token_count", "row"]
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if not header_written:
                writer.writeheader()
                header_written = True
            writer.writerow(item)
        queue.task_done()

def _init_run_context(csv_filename: Path):
    result_queue = asyncio.Queue()
    jobs_queue: asyncio.Queue = asyncio.Queue()
    io_lock = asyncio.Lock()
    counter = {"count": 0}
    writer_task = asyncio.create_task(csv_writer(result_queue, csv_filename))
    return result_queue, jobs_queue, io_lock, counter, writer_task

async def _finalize_run(result_queue: asyncio.Queue, writer_task: asyncio.Task, exc, worker_results):
    await result_queue.put(None)
    await result_queue.join()
    await writer_task
    if exc is not None:
        raise exc
    for r in worker_results:
        if isinstance(r, Exception):
            raise r

def _try_get_prompt_tokens_from_usage(usage: Any) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        v = usage.get("prompt_tokens")
        return int(v) if v is not None else None
    v = getattr(usage, "prompt_tokens", None)
    return int(v) if v is not None else None

async def _run_worker_pool(jobs_queue: asyncio.Queue, *, max_workers: int, process_item: Callable[[Any], Awaitable[None]]):
    stop_event = asyncio.Event()
    first_exc: dict[str, Exception | None] = {"exc": None}

    async def worker():
        while True:
            item = await jobs_queue.get()
            try:
                if item is None:
                    return
                if stop_event.is_set():
                    continue
                await process_item(item)
            except Exception as e:
                if not stop_event.is_set():
                    stop_event.set()
                    first_exc["exc"] = e
            finally:
                jobs_queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(max_workers)]
    for _ in range(max_workers):
        await jobs_queue.put(None)

    await jobs_queue.join()
    worker_results = await asyncio.gather(*workers, return_exceptions=True)
    return first_exc.get("exc"), worker_results

async def process_row(
    idx,
    row,
    queue: asyncio.Queue,
    io_lock: asyncio.Lock,
    counter: dict,
    max_context_window: int,
    *,
    client_api: AsyncOpenAI | None = None,
    model_name: str | None = None,
    samples: int = SAMPLES,
):
    """Process a single dataset row with sampling and retries.
    - For each row, call the API `samples` times and average the grade.
    - Each API call has up to MAX_RETRIES attempts.
    - If a sample fails all retries: the row is skipped (logged, no prompt).
    """
    messages = json.loads(row["prompt"])
    est = n_tokens(messages)
    if est > max_context_window:
        return

    if client_api is None and (client is None or MODEL_CONFIG is None):
        raise RuntimeError("Model is not initialized. Please pass --model-id or call init_model().")

    api_client = client_api if client_api is not None else client
    model = model_name if model_name is not None else MODEL
    model_cfg = MODEL_CONFIG if model == MODEL else load_model_config(model)
    request_kwargs = build_chat_completion_kwargs(model_cfg)

    if client_api is None and model != MODEL:
        api_client = get_client_for_model(model)

    grades: list[float] = []
    last_err_msg = None
    api_prompt_tokens: int | None = None

    for s in range(samples):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                stream = await api_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    timeout=60.0,
                    **request_kwargs,
                    stream=True,
                    stream_options={"include_usage": True},
                )
                chunks: list[str] = []
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    pt = _try_get_prompt_tokens_from_usage(usage)
                    if pt is not None:
                        api_prompt_tokens = pt

                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue

                    choice = choices[0]
                    delta = getattr(choice, "delta", None)
                    if delta is not None:
                        content = getattr(delta, "content", None)
                    else:
                        message = getattr(choice, "message", None)
                        content = getattr(message, "content", None) if message is not None else None
                    if content:
                        chunks.append(content)

                response = "".join(chunks)
                g = grade(response, row["answer"], row["random_string_to_prepend"])
                grades.append(float(g))
                break
            except Exception as e:
                status_code = getattr(e, "status_code", None) or getattr(e, "http_status", None) or "unknown"
                last_err_msg = f"status={status_code}, error={e}"
                continue

        if len(grades) <= s:
            async with io_lock:
                print(
                    f"[SKIP] row={idx} sample={s+1}/{samples} LLM 在 {MAX_RETRIES} 次重试后仍失败，已自动跳过: {last_err_msg}",
                    flush=True,
                )
            return

    avg_g = sum(grades) / len(grades)

    if api_prompt_tokens is None:
        async with io_lock:
            print(
                f"[SKIP CSV] row={idx} samples={samples} avg_grade={avg_g:.6f} "
                f"tiktoken_est={est} api_prompt_tokens=None (usage 未返回 prompt_tokens，结果不写入 CSV)",
                flush=True,
            )
        return

    token_cell = str(api_prompt_tokens)
    async with io_lock:
        counter["count"] += 1
        print(
            f"row={idx} processed={counter['count']} samples={samples} avg_grade={avg_g:.6f} "
            f"tiktoken_est={est} api_prompt_tokens={api_prompt_tokens}",
            flush=True,
        )
    await queue.put({
        "grade": avg_g,
        "token_count": token_cell,
        "row": idx,
    })

async def run_parallel(samples: int, csv_filename: Path | None = None, max_workers: int = 20, max_context_window: int | None = None):
    if dataset is None:
        raise RuntimeError("Dataset is not initialized. Please call load_dataset() first.")
    mcw, max_workers = _run_params(max_context_window, max_workers)
    if csv_filename is None:
        csv_filename = build_default_csv_path(MODEL, needle)
    else:
        csv_filename.parent.mkdir(parents=True, exist_ok=True)
    result_queue, jobs_queue, io_lock, counter, writer_task = _init_run_context(csv_filename)

    for idx, row in dataset.iterrows():
        await jobs_queue.put((idx, row))

    async def process_item(item: Any) -> None:
        idx, row = item
        await process_row(
            idx,
            row,
            result_queue,
            io_lock,
            counter,
            mcw,
            samples=samples,
        )

    exc, worker_results = await _run_worker_pool(jobs_queue, max_workers=max_workers, process_item=process_item)
    await _finalize_run(result_queue, writer_task, exc, worker_results)


# Resume logic: read latest CSV, continue untested rows, and append results
async def run_resume(samples: int, csv_filename: Path, max_workers: int = 20, max_context_window: int | None = None):
    mcw, max_workers = _run_params(max_context_window, max_workers)
    if not csv_filename.exists():
        raise FileNotFoundError(f"未找到结果CSV文件：{csv_filename}")

    print(f"继续测试：文件={csv_filename.name} 模型={csv_filename.parent.name} 数据集={needle}")
    dataset_resume = dataset if dataset is not None else load_dataset(needle)

    tested_rows = set()
    try:
        df_prior = pd.read_csv(csv_filename)
        if "row" in df_prior.columns:
            tested_rows = set(pd.to_numeric(df_prior["row"], errors="coerce").dropna().astype(int).tolist())
    except Exception as e:
        print(f"读取已有结果失败，将从头开始。原因: {e}")

    pending_count = 0
    result_queue, jobs_queue, io_lock, counter, writer_task = _init_run_context(csv_filename)

    for idx, row in dataset_resume.iterrows():
        if idx in tested_rows:
            continue
        pending_count += 1
        await jobs_queue.put((idx, row))

    if pending_count == 0:
        print("CSV 已包含所有行的测试结果，无需续测。")
        await _finalize_run(result_queue, writer_task, None, [])
        return

    print(f"待续测行数: {pending_count}")

    async def process_item(item: Any) -> None:
        idx, row = item
        await process_row(
            idx,
            row,
            result_queue,
            io_lock,
            counter,
            mcw,
            samples=samples,
        )

    exc, worker_results = await _run_worker_pool(jobs_queue, max_workers=max_workers, process_item=process_item)
    await _finalize_run(result_queue, writer_task, exc, worker_results)
    print(f"续测完成，结果已追加到 {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRCR testing and resume")
    parser.add_argument("--save-to", type=str, default=None, help="Path to save CSV results. If file exists, resume mode is used. If not provided, saves to ./results/{needle}/model_timestamp.csv.")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples per question")
    parser.add_argument("--max-workers", type=int, default=20, help="Maximum number of concurrent workers")
    parser.add_argument(
        "--max-context-window",
        type=int,
        default=None,
        help="Max prompt size (tiktoken o200k_base estimate). Rows above this are skipped.",
    )
    parser.add_argument("--needle", type=str, default="8needle", help="Needle dataset subdir name (e.g. 2needle/8needle)")
    parser.add_argument("--model-id", type=str, default="kimi-k2.5", help="Model name (key in models.yaml)")
    args = parser.parse_args()
    samples = max(1, args.samples)
    max_workers = max(1, args.max_workers)
    max_context_window = args.max_context_window if args.max_context_window is not None else default_max_context_window()
    needle = str(args.needle)
    dataset = load_dataset(needle)

    init_model(args.model_id)

    if args.save_to:
        csv_path = Path(args.save_to)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            print(f"文件已存在，使用续测模式：{csv_path}")
            asyncio.run(run_resume(samples, csv_path, max_workers, max_context_window))
        else:
            print(f"保存结果到：{csv_path}")
            asyncio.run(run_parallel(samples, csv_path, max_workers, max_context_window))
    else:
        csv_path = build_default_csv_path(MODEL, needle)
        print(f"保存结果到：{csv_path}")
        asyncio.run(run_parallel(samples, csv_path, max_workers, max_context_window))
