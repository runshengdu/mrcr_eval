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

load_dotenv(override=True)

MAX_CONTEXT_WINDOW = int(128000 * 0.9)
MODEL = "qwen3-30b-a3b-instruct-2507"
needle = "2needle"
CONCURRENCY = 6
SAMPLES = 3

parquet_path = hf_hub_download(repo_id="openai/mrcr", filename=f"{needle}.parquet", repo_type="dataset")
dataset = pd.read_parquet(parquet_path)
client = AsyncOpenAI(api_key=os.environ.get('DASHSCOPE_API_KEY'), base_url=os.environ.get("qwen"))
enc = tiktoken.get_encoding("o200k_base")

def grade(response, answer, random_string_to_prepend) -> float:
    """
    Compare response and answer.
    """
    if not response.startswith(random_string_to_prepend):
        return 0
    response = response.removeprefix(random_string_to_prepend)
    answer = answer.removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())

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
        return parts[0], parts[1]
    return MODEL, needle

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

async def process_row(idx, row, semaphore: asyncio.Semaphore, queue: asyncio.Queue, lock: asyncio.Lock, counter: dict, client_api: AsyncOpenAI = None, model_name: str = None, resume_mode: bool = False):
    """Unified row processing for both fresh-run and resume modes.
    Simplified logic: check API status code first; handle non-200 based on mode.
    """
    messages = json.loads(row["prompt"])
    token_count = n_tokens(messages)
    if token_count > MAX_CONTEXT_WINDOW:
        return

    api_client = client_api if client_api is not None else client
    model = model_name if model_name is not None else MODEL

    async with semaphore:
        try:
            completion = await api_client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30.0
            )
            status_code = getattr(completion, "status_code", 200)
            if status_code != 200:
                err_msg = f"status_code={status_code}, raw={completion}"
                if not resume_mode:
                    async with lock:
                        print(f"[ERROR] row={idx} API返回非200：{err_msg}", flush=True)
                    raise RuntimeError(f"API返回非200：{err_msg}")
                else:
                    async with lock:
                        print(f"[WARN] row={idx} API返回非200：{err_msg}", flush=True)
                        choice = input("返回不是200，要跳过该行吗？(y/n): ").strip().lower()
                    if choice.startswith("y"):
                        async with lock:
                            print(f"[SKIP] row={idx} 用户选择跳过该行。", flush=True)
                        return
                    else:
                        raise RuntimeError(f"用户选择不跳过，终止。详情：{err_msg}")

            response = completion.choices[0].message.content
            g = grade(response, row["answer"], row["random_string_to_prepend"])    
            avg_g = float(g)

        except Exception as e:
            # 异常视为非200，根据模式处理
            status_code = getattr(e, "status_code", None) or getattr(e, "http_status", None) or "unknown"
            err_msg = str(e)
            if not resume_mode:
                async with lock:
                    print(f"[ERROR] row={idx} API异常(视为非200)：status={status_code}, 错误：{err_msg}", flush=True)
                raise
            else:
                async with lock:
                    print(f"[WARN] row={idx} API异常(视为非200)：status={status_code}, 错误：{err_msg}", flush=True)
                    choice = input("返回不是200，要跳过该行吗？(y/n): ").strip().lower()
                if choice.startswith("y"):
                    async with lock:
                        print(f"[SKIP] row={idx} 用户选择跳过该行。", flush=True)
                    return
                else:
                    raise

        async with lock:
            counter["count"] += 1
            print(f"row={idx} processed={counter['count']} avg_grade={avg_g:.6f} tokens={token_count}", flush=True)
        await queue.put({
            "grade": avg_g,
            "token_count": token_count,
            "row": idx,
        })

async def run_parallel():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    lock = asyncio.Lock()
    counter = {"count": 0}
    # Prepare CSV filename early and handle header
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_filename = result_dir / f"{MODEL}_{needle}_{timestamp}.csv"
    writer_task = asyncio.create_task(csv_writer(queue, csv_filename))

    tasks = []
    for idx, row in dataset.iterrows():
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore, queue, lock, counter)))

    await asyncio.gather(*tasks)
    await queue.put(None)
    await queue.join()
    await writer_task


# Resume logic: read latest CSV, continue untested rows, and append results
async def run_resume():
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)
    latest = find_latest_csv(result_dir)
    if latest is None:
        raise FileNotFoundError(f"未找到结果CSV文件，请先运行新测生成：{result_dir}")

    model_resume, needle_resume = parse_model_and_needle(latest.name)
    print(f"继续测试：文件={latest.name} 模型={model_resume} 数据集={needle_resume}")

    parquet_path = hf_hub_download(repo_id="openai/mrcr", filename=f"{needle_resume}.parquet", repo_type="dataset")
    dataset_resume = pd.read_parquet(parquet_path)

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
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore, queue, lock, counter, client_resume, None, True)))

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
    parser = argparse.ArgumentParser(description="MRCR 测试与续测")
    parser.add_argument("--resume", action="store_true", default=False, help="是否进行续测（从最新CSV继续）")
    args = parser.parse_args()
    if args.resume:
        asyncio.run(run_resume())
    else:
        asyncio.run(run_parallel())
