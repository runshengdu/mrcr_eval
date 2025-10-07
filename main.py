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

load_dotenv()

MAX_CONTEXT_WINDOW = int(128000 * 0.9)
MODEL= "qwen3-next-80b-a3b-instruct"
needle="4needle"
CONCURRENCY = 5
SAMPLES = 3

parquet_path = hf_hub_download(repo_id="openai/mrcr", filename=f"{needle}.parquet", repo_type="dataset")
dataset = pd.read_parquet(parquet_path)
api_key = os.environ.get('DASHSCOPE_API_KEY') 
client = AsyncOpenAI(api_key=api_key, base_url=os.environ.get('qwen'))
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

def n_tokens(messages : list[dict]) -> int:
    """
    Count tokens in messages.
    """
    return sum([len(enc.encode(m["content"])) for m in messages])

# Prepare CSV filename early and handle header
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
n_needle = needle

# Ensure output directory exists and set CSV path to result/
result_dir = Path("result")
result_dir.mkdir(parents=True, exist_ok=True)
csv_filename = result_dir / f"{MODEL}_{n_needle}_{timestamp}.csv"

# Ensure logs directory exists and set JSON log path
logs_dir = Path("logs")
logs_dir.mkdir(parents=True, exist_ok=True)
json_log_filename = logs_dir / f"{MODEL}_{n_needle}_{timestamp}.json"


async def csv_writer(queue: asyncio.Queue, filename: str):
    header_written = os.path.exists(filename)
    fieldnames = ["grade", "token_count"]
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

async def json_writer(queue: asyncio.Queue, filename: str):
    # Write a JSON array with comma-separated objects
    file_exists = os.path.exists(filename)
    first_item = True
    if not file_exists:
        with open(filename, "a", encoding="utf-8") as f:
            f.write("[\n")
    while True:
        item = await queue.get()
        if item is None:
            # close the JSON array
            with open(filename, "a", encoding="utf-8") as f:
                f.write("\n]\n")
            queue.task_done()
            break
        with open(filename, "a", encoding="utf-8") as f:
            if not first_item:
                f.write(",\n")
            f.write(json.dumps(item, ensure_ascii=False))
            first_item = False
        queue.task_done()

async def process_row(idx, row, semaphore: asyncio.Semaphore, queue: asyncio.Queue, lock: asyncio.Lock, counter: dict, log_queue: asyncio.Queue):
    messages = json.loads(row["prompt"])
    token_count = n_tokens(messages)
    if token_count > MAX_CONTEXT_WINDOW:
        return
    async with semaphore:
        grades = []
        for _ in range(SAMPLES):
            retries = 3
            delay = 1.0
            for attempt in range(retries):
                try:
                    completion = await client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                    )
                    response = completion.choices[0].message.content
                    g = grade(response, row["answer"], row["random_string_to_prepend"])    
                    grades.append(g)
                    # Log LLM input, output, and grade into JSON logs
                    await log_queue.put({
                        "input": messages,
                        "output": response,
                        "grade": g,
                    })
                    break
                except Exception:
                    if attempt == retries - 1:
                        # give up this sample, proceed to next
                        break
                    await asyncio.sleep(delay)
                    delay *= 2

        max_g = float(max(grades)) if grades else 0.0
        async with lock:
            counter["count"] += 1
            print(f"row={idx} processed={counter['count']} max_grade={max_g:.6f} tokens={token_count}")
        await queue.put({
            "grade": max_g,
            "token_count": token_count,
        })
        return

async def run_parallel():
    semaphore = asyncio.Semaphore(CONCURRENCY)
    queue = asyncio.Queue()
    log_queue = asyncio.Queue()
    lock = asyncio.Lock()
    counter = {"count": 0}
    writer_task = asyncio.create_task(csv_writer(queue, csv_filename))
    log_writer_task = asyncio.create_task(json_writer(log_queue, json_log_filename))

    tasks = []
    for idx, row in dataset.iterrows():
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore, queue, lock, counter, log_queue)))

    await asyncio.gather(*tasks)
    await queue.put(None)
    await queue.join()
    await writer_task
    await log_queue.put(None)
    await log_queue.join()
    await log_writer_task

asyncio.run(run_parallel())
