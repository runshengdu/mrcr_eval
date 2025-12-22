import argparse
from huggingface_hub import hf_hub_download
import pandas as pd
from openai import AsyncOpenAI
import json
from difflib import SequenceMatcher
import tiktoken
import os
from datetime import datetime
from pathlib import Path
import asyncio
import csv
from dotenv import load_dotenv
import re

load_dotenv()

MAX_CONTEXT_WINDOW = int(200000 * 0.85)
MODEL = "google/gemini-3-flash-preview"
needle = "2needle"
CONCURRENCY = 30
SAMPLES = 1
MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 0

dataset = pd.concat([
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
client = AsyncOpenAI(api_key=os.environ.get("OPENROUTER_API_KEY"), base_url=os.environ.get("OPENROUTER"))
enc = tiktoken.get_encoding("o200k_base")

def grade(response, answer, random_string_to_prepend) -> float:
    """
    Compare response and answer with strict prefix enforcement.
    """
    # Strip think tags from response before prefix check
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)

    # Normalize leading BOM/zero-width/whitespace before prefix check
    def normalize_leading(s: str) -> str:
        return re.sub(r"^[\ufeff\u200b\u200c\u200d\s]+", "", s.replace("\r\n", "\n"))

    response_norm = normalize_leading(response)
    if not response_norm.startswith(random_string_to_prepend):
        return 0.0

    response_body = response_norm.removeprefix(random_string_to_prepend).strip()
    answer_body = normalize_leading(answer).removeprefix(random_string_to_prepend).strip()
    return float(SequenceMatcher(None, response_body, answer_body).ratio())

def n_tokens(messages: list[dict]) -> int:
    """
    Count tokens in messages.
    """
    return sum([len(enc.encode(m["content"])) for m in messages])

def find_latest_csv(result_dir: Path) -> Path | None:
    files = list(result_dir.glob("*.csv"))
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]

def parse_model_and_needle(filename: str) -> tuple[str, str]:
    name = Path(filename).name.replace(".csv", "")
    parts = name.split("_")
    if len(parts) >= 2:
        return parts[0]
    return MODEL

async def csv_writer(queue: asyncio.Queue, filename: str):
    header_written = os.path.exists(filename)
    fieldnames = ["grade", "token_count", "row"]
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not header_written:
                writer.writeheader() 
                header_written = True
            writer.writerow(item)
        queue.task_done()

async def process_row(idx, row, semaphore: asyncio.Semaphore, queue: asyncio.Queue, lock: asyncio.Lock, counter: dict, client_api: AsyncOpenAI = None, model_name: str = None, resume_mode: bool = False, samples: int = SAMPLES):
    """Process a single dataset row with sampling and retries.
    - For each row, call the API `samples` times and average the grade.
    - Each API call has up to MAX_RETRIES attempts.
    - If a sample fails all retries: non-resume mode aborts; resume mode asks whether to skip the row.
    """
    messages = json.loads(row["prompt"])
    token_count = n_tokens(messages)
    if token_count > MAX_CONTEXT_WINDOW:
        return

    api_client = client_api if client_api is not None else client
    model = model_name if model_name is not None else MODEL

    async with semaphore:
        grades: list[float] = []
        for s in range(samples):
            last_err_msg = None
            # retry up to MAX_RETRIES for this sample
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)
                    extra_body = {"enable_thinking": True}
                    if model == "openai/gpt-5.2":
                        extra_body["reasoning_effort"] = "high"
                    stream = await api_client.chat.completions.create(
                        model=model,
                        messages=messages,
                        timeout=30.0,
                        extra_body=extra_body,
                        stream=True,
                    )
                    chunks: list[str] = []
                    async for chunk in stream:
                        # OpenAI-compatible streaming: each chunk has choices[0].delta.content
                        choice = chunk.choices[0]
                        delta = getattr(choice, "delta", None)
                        content = getattr(delta, "content", None) if delta is not None else None
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

            # after retries, if this sample didn't succeed
            if len(grades) <= s:
                async with lock:
                    print(f"[ERROR] row={idx} sample={s+1}/{samples} failed after {MAX_RETRIES} retries: {last_err_msg}", flush=True)
                    if resume_mode:
                        choice = input("该题在所有重试后仍失败，是否跳过该行？(y/n): ").strip().lower()
                    else:
                        choice = None
                if resume_mode:
                    if choice and choice.startswith("y"):
                        async with lock:
                            print(f"[SKIP] row={idx} 用户选择跳过该行。", flush=True)
                        return
                    else:
                        raise RuntimeError(f"row={idx} failed after retries: {last_err_msg}")
                else:
                    raise RuntimeError(f"row={idx} failed after retries: {last_err_msg}")
        avg_g = sum(grades) / len(grades) if grades else 0.0
        async with lock:
            counter["count"] += 1
            print(f"row={idx} processed={counter['count']} samples={samples} avg_grade={avg_g:.6f} tokens={token_count}", flush=True)
        await queue.put({
            "grade": avg_g,
            "token_count": token_count,
            "row": idx,
        })

async def run_parallel(samples: int):
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    lock = asyncio.Lock()
    counter = {"count": 0}
    # Prepare CSV filename early and handle header
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    def _safe_component(s: str) -> str:
        # Replace path separators and illegal filename characters with '_'
        # Covers '/', '\\', ':', '*', '?', '"', '<', '>', '|'
        return re.sub(r"[\\/:*?\"<>|]+", "_", s)

    safe_model = _safe_component(MODEL)
    safe_needle = _safe_component(needle)
    csv_filename = result_dir / f"{safe_model}_{timestamp}.csv"
    writer_task = asyncio.create_task(csv_writer(queue, csv_filename))

    tasks = []
    for idx, row in dataset.iterrows():
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore, queue, lock, counter, resume_mode=False, samples=samples)))

    await asyncio.gather(*tasks)
    await queue.put(None)
    await queue.join()
    await writer_task


# Resume logic: read latest CSV, continue untested rows, and append results
async def run_resume(samples: int):
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    latest = find_latest_csv(result_dir)
    if latest is None:
        raise FileNotFoundError(f"未找到结果CSV文件，请先运行新测生成：{result_dir}")

    model_resume = parse_model_and_needle(latest.name)
    print(f"继续测试：文件={latest.name} 模型={model_resume} 数据集={needle}")

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
    try:
        df_done = pd.read_csv(latest)
        if "row" in df_done.columns:
            tested_rows = set(pd.to_numeric(df_done["row"], errors="coerce").dropna().astype(int).tolist())
    except Exception as e:
        print(f"读取已有结果失败，将从头开始。原因: {e}")

    # Use the global client and global MODEL; do not switch client by filename
    client_resume = client

    pending_count = 0
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    lock = asyncio.Lock()
    counter = {"count": 0}
    writer_task = asyncio.create_task(csv_writer(queue, str(latest)))

    tasks = []
    for idx, row in dataset_resume.iterrows():
        if idx in tested_rows:
            continue
        pending_count += 1
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore, queue, lock, counter, client_resume, None, True, samples)))

    if not tasks:
        print("CSV 已包含所有行的测试结果，无需续测。")
        await queue.put(None)
        await queue.join()
        await writer_task
        return

    print(f"待续测行数: {pending_count}")
    await asyncio.gather(*tasks)
    await queue.put(None)
    await queue.join()
    await writer_task
    print(f"续测完成，结果已追加到 {latest}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRCR testing and resume")
    parser.add_argument("--resume", type=str.lower, choices=["true", "false"], default="false", help="Resume from latest CSV (true/false)")
    parser.add_argument("--samples", type=int, default=SAMPLES, help="Number of samples per question")
    args = parser.parse_args()
    resume_mode = args.resume == "true"
    samples = max(1, args.samples)
    if resume_mode:
        asyncio.run(run_resume(samples))
    else:
        asyncio.run(run_parallel(samples))
