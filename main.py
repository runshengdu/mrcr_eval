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
from typing import Any
import yaml

MAX_CONTEXT_WINDOW = int(200000*0.9)
MODEL = ""
needle = ""
SAMPLES = 1
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 0
TOKEN_RATIO_UPDATE_EVERY = 10
WARMUP_MAX_PROMPT_CHARS = 50000


_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


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


def get_client_for_model(model_name: str) -> AsyncOpenAI:
    cfg = load_model_config(model_name)
    cache_key = (str(cfg["base_url"]), str(cfg["api_key"]))
    cached = _CLIENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    new_client = AsyncOpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
    _CLIENT_CACHE[cache_key] = new_client
    return new_client


_CHAT_COMPLETION_CONFIG_KEYS = {
    "temperature",
    "max_tokens",
    "extra_body",
}


def build_chat_completion_kwargs(model_cfg: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    for k in _CHAT_COMPLETION_CONFIG_KEYS:
        if k in model_cfg and model_cfg[k] is not None:
            kwargs[k] = model_cfg[k]
    return kwargs

MODEL_CONFIG: dict[str, Any] | None = None
client: AsyncOpenAI | None = None

def init_model(model_id: str) -> None:
    global MODEL, MODEL_CONFIG, client
    MODEL = str(model_id)
    MODEL_CONFIG = load_model_config(MODEL)
    client = AsyncOpenAI(api_key=MODEL_CONFIG["api_key"], base_url=MODEL_CONFIG["base_url"])

def load_dataset(selected_needle: str) -> pd.DataFrame:
    selected_needle = str(selected_needle)
    return pd.concat([
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename=f"{selected_needle}/{selected_needle}_0.parquet",
                repo_type="dataset",
            )
        ),
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename=f"{selected_needle}/{selected_needle}_1.parquet",
                repo_type="dataset",
            )
        ),
    ])

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

def prompt_char_count(messages: list[dict]) -> int:
    total = 0
    for m in messages:
        c = m.get("content", "")
        if c is None:
            continue
        total += len(str(c))
    return total

def row_prompt_char_count(row) -> int | None:
    try:
        messages = json.loads(row["prompt"])
        return prompt_char_count(messages)
    except Exception:
        return None

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
        fieldnames = ["grade", "token_count", "row", "prompt_char_count", "real_prompt_tokens", "updated_token_ratio"]
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

def _try_get_prompt_tokens_from_usage(usage: Any) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        v = usage.get("prompt_tokens")
        return int(v) if v is not None else None
    v = getattr(usage, "prompt_tokens", None)
    return int(v) if v is not None else None

def _latest_ratio_from_existing_csv(df_done: pd.DataFrame) -> float | None:
    if df_done is None or df_done.empty:
        return None
    if "updated_token_ratio" in df_done.columns:
        s = pd.to_numeric(df_done["updated_token_ratio"], errors="coerce").dropna()
        if not s.empty:
            return float(s.iloc[-1])
    if "real_prompt_tokens" in df_done.columns and "prompt_char_count" in df_done.columns:
        t = pd.to_numeric(df_done["real_prompt_tokens"], errors="coerce")
        c = pd.to_numeric(df_done["prompt_char_count"], errors="coerce")
        ok = (~t.isna()) & (~c.isna()) & (c > 0)
        if bool(ok.any()):
            last_t = float(t[ok].iloc[-1])
            last_c = float(c[ok].iloc[-1])
            return float(last_t / last_c)
    return None

async def _create_stream_with_usage(api_client: AsyncOpenAI, *, model: str, messages: list[dict], timeout: float, request_kwargs: dict[str, Any]):
    try:
        return await api_client.chat.completions.create(
            model=model,
            messages=messages,
            timeout=timeout,
            **request_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        )
    except Exception:
        raise

async def _get_current_ratio(ratio_state: dict[str, Any], ratio_lock: asyncio.Lock) -> float | None:
    async with ratio_lock:
        return ratio_state.get("ratio")

async def _observe_ratio(ratio_state: dict[str, Any], ratio_lock: asyncio.Lock, observed_ratio: float) -> float:
    async with ratio_lock:
        current = ratio_state.get("ratio")
        if current is None:
            ratio_state["ratio"] = float(observed_ratio)
            ratio_state["observed_sum"] = float(observed_ratio)
            ratio_state["observed_count"] = 1
            ratio_state["updates"] = 0
            return ratio_state["ratio"]

        observed_sum = float(ratio_state.get("observed_sum") or 0.0) + float(observed_ratio)
        observed_count = int(ratio_state.get("observed_count") or 0) + 1
        ratio_state["observed_sum"] = observed_sum
        ratio_state["observed_count"] = observed_count

        if observed_count % TOKEN_RATIO_UPDATE_EVERY == 0:
            ratio_state["ratio"] = float(observed_sum / float(observed_count))
            ratio_state["updates"] = int(ratio_state.get("updates") or 0) + 1
        return float(ratio_state["ratio"])

async def process_row(idx, row, queue: asyncio.Queue, io_lock: asyncio.Lock, ratio_state: dict[str, Any], ratio_lock: asyncio.Lock, counter: dict, client_api: AsyncOpenAI = None, model_name: str = None, resume_mode: bool = False, samples: int = SAMPLES, skip_length_check: bool = False, emit_result: bool = True):
    """Process a single dataset row with sampling and retries.
    - For each row, call the API `samples` times and average the grade.
    - Each API call has up to MAX_RETRIES attempts.
    - If a sample fails all retries: non-resume mode aborts; resume mode asks whether to skip the row.
    """
    messages = json.loads(row["prompt"])
    char_count = prompt_char_count(messages)
    current_ratio = await _get_current_ratio(ratio_state, ratio_lock)
    # If ratio is not initialized (Warmup), we use 0 (and skip check is True)
    # If ratio is initialized, we use char_count * ratio
    estimated_token_count = int(char_count * current_ratio) if (current_ratio is not None) else 0
    
    
    if not skip_length_check and estimated_token_count > MAX_CONTEXT_WINDOW:
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
    row_ratio_observed = False
    real_prompt_tokens: int | None = None
    updated_ratio: float | None = None

    for s in range(samples):
        # retry up to MAX_RETRIES for this sample
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                await asyncio.sleep(REQUEST_DELAY_SECONDS)
                stream = await _create_stream_with_usage(
                    api_client,
                    model=model,
                    messages=messages,
                    timeout=60.0,
                    request_kwargs=request_kwargs,
                )
                chunks: list[str] = []
                async for chunk in stream:
                    usage = getattr(chunk, "usage", None)
                    pt = _try_get_prompt_tokens_from_usage(usage)
                    if pt is not None:
                        real_prompt_tokens = pt

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

                if (not row_ratio_observed) and (real_prompt_tokens is not None) and char_count > 0:
                    observed_ratio = float(real_prompt_tokens) / float(char_count)
                    updated_ratio = await _observe_ratio(ratio_state, ratio_lock, observed_ratio)
                    row_ratio_observed = True
                break
            except Exception as e:
                status_code = getattr(e, "status_code", None) or getattr(e, "http_status", None) or "unknown"
                last_err_msg = f"status={status_code}, error={e}"
                continue

        # after retries, if this sample didn't succeed
        if len(grades) <= s:
            async with io_lock:
                print(f"[ERROR] row={idx} sample={s+1}/{samples} failed after {MAX_RETRIES} retries: {last_err_msg}", flush=True)
                if resume_mode:
                    choice = input("该题在所有重试后仍失败，是否跳过该行？(y/n): ").strip().lower()
                else:
                    choice = None
            if resume_mode:
                if choice and choice.startswith("y"):
                    async with io_lock:
                        print(f"[SKIP] row={idx} 用户选择跳过该行。", flush=True)
                    return
                raise RuntimeError(f"row={idx} failed after retries: {last_err_msg}")
            raise RuntimeError(f"row={idx} failed after retries: {last_err_msg}")

    avg_g = sum(grades) / len(grades) if grades else 0.0
    
    # Final token count for CSV/Logging: prefer real, then estimated
    final_token_count = real_prompt_tokens if real_prompt_tokens is not None else estimated_token_count

    if emit_result:
        async with io_lock:
            counter["count"] += 1
            ratio_for_print = updated_ratio if updated_ratio is not None else current_ratio
            ratio_str = f"{ratio_for_print:.8f}" if ratio_for_print is not None else "none"
            print(
                f"row={idx} processed={counter['count']} samples={samples} avg_grade={avg_g:.6f} "
                f"tokens={real_prompt_tokens} chars={char_count} ratio={ratio_str}",
                flush=True,
            )
        await queue.put({
            "grade": avg_g,
            "token_count": final_token_count,
            "row": idx,
            "prompt_char_count": char_count,
            "real_prompt_tokens": real_prompt_tokens,
            "updated_token_ratio": updated_ratio,
        })

async def run_parallel(samples: int, csv_filename: Path = None, max_workers: int = 20):
    if dataset is None:
        raise RuntimeError("Dataset is not initialized. Please call load_dataset() first.")
    max_workers = max(1, int(max_workers))
    result_queue = asyncio.Queue()
    jobs_queue: asyncio.Queue = asyncio.Queue()
    io_lock = asyncio.Lock()
    ratio_lock = asyncio.Lock()
    ratio_state: dict[str, Any] = {"ratio": None, "observed_sum": 0.0, "observed_count": 0, "updates": 0}
    counter = {"count": 0}
    # Prepare CSV filename early and handle header
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        def _safe_component(s: str) -> str:
            # Replace path separators and illegal filename characters with '_'
            # Covers '/', '\\', ':', '*', '?', '"', '<', '>', '|'
            return re.sub(r"[\\/:*?\"<>|]+", "_", s)

        safe_model = _safe_component(MODEL)
        safe_needle = _safe_component(needle)
        result_dir = Path("results") / safe_needle
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_filename = result_dir / f"{safe_model}_{timestamp}.csv"
    else:
        csv_filename.parent.mkdir(parents=True, exist_ok=True)
    writer_task = asyncio.create_task(csv_writer(result_queue, csv_filename))

    warmup_attempts = 0
    MAX_WARMUP_ATTEMPTS = 3

    for idx, row in dataset.iterrows():
        if warmup_attempts >= MAX_WARMUP_ATTEMPTS:
            break
        cc = row_prompt_char_count(row)
        if cc is None or cc >= WARMUP_MAX_PROMPT_CHARS:
            continue

        warmup_attempts += 1
        print(f"Running warmup on row={idx} (attempt {warmup_attempts}/{MAX_WARMUP_ATTEMPTS})")

        try:
            await process_row(
                idx,
                row,
                result_queue,
                io_lock,
                ratio_state,
                ratio_lock,
                counter,
                resume_mode=False,
                samples=samples,
                skip_length_check=True,
                emit_result=False,
            )
        except Exception as e:
            print(f"Warmup failed on row={idx}: {e}")

        if (await _get_current_ratio(ratio_state, ratio_lock)) is not None:
            print(f"Warmup success. Ratio initialized.")
            break

    if warmup_attempts == 0:
        for idx, row in dataset.iterrows():
            if warmup_attempts >= MAX_WARMUP_ATTEMPTS:
                break

            warmup_attempts += 1
            print(f"Running warmup on row={idx} (attempt {warmup_attempts}/{MAX_WARMUP_ATTEMPTS})")

            try:
                await process_row(
                    idx,
                    row,
                    result_queue,
                    io_lock,
                    ratio_state,
                    ratio_lock,
                    counter,
                    resume_mode=False,
                    samples=samples,
                    skip_length_check=True,
                    emit_result=False,
                )
            except Exception as e:
                print(f"Warmup failed on row={idx}: {e}")

            if (await _get_current_ratio(ratio_state, ratio_lock)) is not None:
                print(f"Warmup success. Ratio initialized.")
                break

    if (await _get_current_ratio(ratio_state, ratio_lock)) is None:
        raise RuntimeError(f"Warmup failed: Could not initialize token ratio after {warmup_attempts} attempts.")

    for idx, row in dataset.iterrows():
        await jobs_queue.put((idx, row))

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
                idx, row = item
                await process_row(
                    idx,
                    row,
                    result_queue,
                    io_lock,
                    ratio_state,
                    ratio_lock,
                    counter,
                    resume_mode=False,
                    samples=samples,
                )
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

    await result_queue.put(None)
    await result_queue.join()
    await writer_task

    exc = first_exc.get("exc")
    if exc is not None:
        raise exc
    for r in worker_results:
        if isinstance(r, Exception):
            raise r


# Resume logic: read latest CSV, continue untested rows, and append results
async def run_resume(samples: int, csv_filename: Path, max_workers: int = 20):
    max_workers = max(1, int(max_workers))
    if not csv_filename.exists():
        raise FileNotFoundError(f"未找到结果CSV文件：{csv_filename}")

    model_resume = csv_filename.parent.name
    print(f"继续测试：文件={csv_filename.name} 模型={model_resume} 数据集={needle}")

    dataset_resume = pd.concat([
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename=f"{needle}/{needle}_0.parquet",
                repo_type="dataset",
            )
        ),
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename=f"{needle}/{needle}_1.parquet",
                repo_type="dataset",
            )
        ),
    ])

    tested_rows = set()
    df_done = None
    try:
        df_done = pd.read_csv(csv_filename)
        if "row" in df_done.columns:
            tested_rows = set(pd.to_numeric(df_done["row"], errors="coerce").dropna().astype(int).tolist())
    except Exception as e:
        print(f"读取已有结果失败，将从头开始。原因: {e}")

    # Use the global client and global MODEL; do not switch client by filename
    client_resume = client

    pending_count = 0
    result_queue = asyncio.Queue()
    jobs_queue: asyncio.Queue = asyncio.Queue()
    io_lock = asyncio.Lock()
    ratio_lock = asyncio.Lock()
    ratio_state: dict[str, Any] = {"ratio": None, "observed_sum": 0.0, "observed_count": 0, "updates": 0}
    counter = {"count": 0}
    writer_task = asyncio.create_task(csv_writer(result_queue, csv_filename))

    existing_ratio = _latest_ratio_from_existing_csv(df_done) if df_done is not None else None
    if existing_ratio is None:
        raise RuntimeError("Resume failed: Could not load token ratio from existing CSV (updated_token_ratio/real_prompt_tokens).")
    async with ratio_lock:
        ratio_state["ratio"] = float(existing_ratio)
        ratio_state["observed_sum"] = float(existing_ratio)
        ratio_state["observed_count"] = 1
        ratio_state["updates"] = 0

    for idx, row in dataset_resume.iterrows():
        if idx in tested_rows:
            continue
        pending_count += 1
        await jobs_queue.put((idx, row))

    if pending_count == 0:
        print("CSV 已包含所有行的测试结果，无需续测。")
        await result_queue.put(None)
        await result_queue.join()
        await writer_task
        return

    print(f"待续测行数: {pending_count}")

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
                idx, row = item
                await process_row(
                    idx,
                    row,
                    result_queue,
                    io_lock,
                    ratio_state,
                    ratio_lock,
                    counter,
                    client_resume,
                    None,
                    True,
                    samples,
                )
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

    await result_queue.put(None)
    await result_queue.join()
    await writer_task

    exc = first_exc.get("exc")
    if exc is not None:
        raise exc
    for r in worker_results:
        if isinstance(r, Exception):
            raise r
    print(f"续测完成，结果已追加到 {csv_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRCR testing and resume")
    parser.add_argument("--save-to", type=str, default=None, help="Path to save CSV results. If file exists, resume mode is used. If not provided, saves to ./results/{needle}/model_timestamp.csv.")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples per question")
    parser.add_argument("--max-workers", type=int, default=20, help="Maximum number of concurrent workers")
    parser.add_argument("--needle", type=str, default="8needle", help="Needle dataset subdir name (e.g. 2needle/8needle)")
    parser.add_argument("--model-id", type=str, default="kimi-k2.5", help="Model name (key in models.yaml)")
    args = parser.parse_args()
    samples = max(1, args.samples)
    max_workers = max(1, args.max_workers)
    needle = str(args.needle)
    dataset = load_dataset(needle)

    init_model(args.model_id)
    
    if args.save_to:
        csv_path = Path(args.save_to)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.exists():
            print(f"文件已存在，使用续测模式：{csv_path}")
            asyncio.run(run_resume(samples, csv_path, max_workers))
        else:
            print(f"保存结果到：{csv_path}")
            asyncio.run(run_parallel(samples, csv_path, max_workers))
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        def _safe_component(s: str) -> str:
            return re.sub(r"[\\/:*?\"<>|]+", "_", s)
        safe_model = _safe_component(MODEL)
        safe_needle = _safe_component(needle)
        result_dir = Path("results") / safe_needle
        result_dir.mkdir(parents=True, exist_ok=True)
        csv_path = result_dir / f"{safe_model}_{timestamp}.csv"
        print(f"保存结果到：{csv_path}")
        asyncio.run(run_parallel(samples, csv_path, max_workers))
