import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

# Ensure repository root is importable when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as mrcr_main

TERMINAL_BATCH_STATES = {"completed", "failed", "expired", "cancelled"}
KIMI_BATCH_FORBIDDEN_PARAMS = {
    "temperature",
    "max_tokens",
    "top_p",
    "n",
    "presence_penalty",
    "frequency_penalty",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MRCR with Moonshot Batch API")
    parser.add_argument(
        "--step",
        type=str,
        default="all",
        choices=["all", "prepare", "upload", "create", "wait", "collect", "submit", "poll"],
        help="Pipeline step to run. Recommended flow: prepare/upload/create/wait/collect.",
    )
    parser.add_argument("--needle", type=str, default="8needle", help="Needle dataset subdir name")
    parser.add_argument("--model-id", type=str, default="kimi-k2.6", help="Model name in models.yaml")
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
        help="CSV path for results. If the file exists, resume mode is used.",
    )
    parser.add_argument(
        "--max-context-window",
        type=int,
        default=(256000 * 0.9),
        help="Max prompt size (tiktoken o200k_base estimate). Rows above this are skipped.",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default="24h",
        help="Batch completion window. Default is 24h.",
    )
    parser.add_argument(
        "--poll-interval-seconds",
        type=int,
        default=10,
        help="Batch status polling interval in seconds.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="batch_api/moonshot/artifacts",
        help="Directory to store intermediate artifacts (jsonl, metadata).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Existing run directory, required by upload/create/wait/collect steps.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Batch ID override for wait/collect steps.",
    )
    return parser.parse_args()


def make_client(model_name: str) -> OpenAI:
    cfg = mrcr_main.load_model_config(model_name)
    return OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])


def read_tested_rows(csv_filename: Path) -> set[int]:
    if not csv_filename.exists():
        return set()
    try:
        df_prior = pd.read_csv(csv_filename)
    except Exception as exc:
        print(f"读取已有 CSV 失败，将视为无历史结果: {exc}")
        return set()
    if "row" not in df_prior.columns:
        return set()
    return set(pd.to_numeric(df_prior["row"], errors="coerce").dropna().astype(int).tolist())


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_run_dir(artifacts_dir: Path, model_name: str, needle: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = mrcr_main.safe_filename_component(model_name)
    safe_needle = mrcr_main.safe_filename_component(needle)
    run_dir = artifacts_dir / safe_needle / safe_model / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_batch_input_file(
    dataset: pd.DataFrame,
    pending_rows: set[int],
    model_name: str,
    max_context_window: int,
    input_path: Path,
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    row_payloads: dict[int, dict[str, Any]] = {}
    skipped_context: set[int] = set()
    model_cfg = mrcr_main.load_model_config(model_name)
    model_kwargs = mrcr_main.build_chat_completion_kwargs(model_cfg)

    with open(input_path, "w", encoding="utf-8", newline="\n") as f:
        for idx, row in dataset.iterrows():
            row_idx = int(idx)
            if row_idx not in pending_rows:
                continue

            messages = json.loads(row["prompt"])
            est_tokens = mrcr_main.n_tokens(messages)
            if est_tokens > max_context_window:
                skipped_context.add(row_idx)
                continue

            request_obj = {
                "custom_id": f"row-{row_idx}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": build_batch_request_body(model_name, messages, model_kwargs),
            }
            f.write(json.dumps(request_obj, ensure_ascii=False) + "\n")
            row_payloads[row_idx] = {
                "answer": row["answer"],
                "random_string_to_prepend": row["random_string_to_prepend"],
                "est_tokens": est_tokens,
            }

    return row_payloads, skipped_context


def extract_text_from_file_content(content_obj: Any) -> str:
    text_attr = getattr(content_obj, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    if callable(text_attr):
        return text_attr()
    read_method = getattr(content_obj, "read", None)
    if callable(read_method):
        raw = read_method()
        if isinstance(raw, bytes):
            return raw.decode("utf-8")
        return str(raw)
    return str(content_obj)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def build_batch_request_body(
    model_name: str,
    messages: list[dict[str, Any]],
    model_kwargs: dict[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
    }
    kwargs = dict(model_kwargs)
    extra_body = kwargs.pop("extra_body", None)
    body.update(kwargs)
    if isinstance(extra_body, dict):
        body.update(extra_body)

    if model_name in {"kimi-k2.5", "kimi-k2.6"}:
        for k in KIMI_BATCH_FORBIDDEN_PARAMS:
            body.pop(k, None)
    return body


def resolve_csv_path(args: argparse.Namespace, model_name: str, needle: str) -> Path:
    if args.save_to:
        csv_path = Path(args.save_to)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        return csv_path
    return mrcr_main.build_default_csv_path(model_name, needle)


def upload_input_file(client: OpenAI, input_path: Path) -> str:
    with open(input_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"输入文件已上传: {file_obj.id}")
    return file_obj.id


def create_batch(client: OpenAI, input_file_id: str, completion_window: str) -> str:
    batch = client.batches.create(
        input_file_id=input_file_id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    print(f"Batch 任务已创建: {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str, poll_interval_seconds: int) -> Any:
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = getattr(batch, "request_counts", None)
        completed = getattr(counts, "completed", 0) if counts else 0
        total = getattr(counts, "total", 0) if counts else 0
        print(f"状态: {batch.status} ({completed}/{total})")

        if batch.status in TERMINAL_BATCH_STATES:
            return batch
        time.sleep(max(1, int(poll_interval_seconds)))


def parse_row_idx(custom_id: str) -> int | None:
    if not isinstance(custom_id, str):
        return None
    if not custom_id.startswith("row-"):
        return None
    value = custom_id[4:]
    if not value.isdigit():
        return None
    return int(value)


def parse_batch_output(
    output_text: str,
    row_payloads: dict[int, dict[str, Any]],
) -> tuple[dict[int, dict[str, Any]], set[int]]:
    row_results: dict[int, dict[str, Any]] = {}
    failed_rows: set[int] = set()

    for raw_line in output_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue

        row_idx = parse_row_idx(data.get("custom_id"))
        if row_idx is None or row_idx not in row_payloads:
            continue

        error = data.get("error")
        response = data.get("response") or {}
        status_code = response.get("status_code")
        body = response.get("body") or {}

        # Moonshot batch output uses status_code=0 for successful rows.
        # Keep compatibility with OpenAI-style status_code=200 as well.
        if error is not None or status_code not in {0, 200}:
            failed_rows.add(row_idx)
            continue

        choices = body.get("choices") or []
        if not choices:
            failed_rows.add(row_idx)
            continue

        message = choices[0].get("message") or {}
        content = message.get("content")
        usage = body.get("usage") or {}
        prompt_tokens = usage.get("prompt_tokens")

        if not isinstance(content, str) or prompt_tokens is None:
            failed_rows.add(row_idx)
            continue

        payload = row_payloads[row_idx]
        g = mrcr_main.grade(content, payload["answer"], payload["random_string_to_prepend"])
        row_results[row_idx] = {
            "grade": float(g),
            "token_count": str(int(prompt_tokens)),
            "row": row_idx,
        }

    return row_results, failed_rows


def append_results_csv(csv_filename: Path, rows: list[dict[str, Any]]) -> None:
    existing = mrcr_main._read_existing_csv_header(csv_filename)
    if existing:
        fieldnames = existing
        write_header = False
    else:
        fieldnames = ["grade", "token_count", "row"]
        write_header = True

    csv_filename.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for item in rows:
            writer.writerow(item)


def stage_prepare(args: argparse.Namespace) -> Path:
    needle = str(args.needle)
    model_name = str(args.model_id)
    max_context_window = (
        int(args.max_context_window)
        if args.max_context_window is not None
        else mrcr_main.default_max_context_window()
    )
    dataset = mrcr_main.load_dataset(needle)
    csv_path = resolve_csv_path(args, model_name, needle)

    tested_rows = read_tested_rows(csv_path)
    all_rows = set(int(idx) for idx in dataset.index.tolist())
    pending_rows = all_rows - tested_rows

    if not pending_rows:
        print("CSV 已包含所有行的测试结果，无需续测。")
        raise SystemExit(0)

    artifacts_root = Path(args.artifacts_dir)
    run_dir = build_run_dir(artifacts_root, model_name, needle)
    input_jsonl = run_dir / "batch_input.jsonl"
    output_jsonl = run_dir / "batch_output.jsonl"
    error_jsonl = run_dir / "batch_error.jsonl"
    row_payloads_json = run_dir / "row_payloads.json"
    meta_json = run_dir / "meta.json"

    row_payloads, skipped_context_rows = build_batch_input_file(
        dataset=dataset,
        pending_rows=pending_rows,
        model_name=model_name,
        max_context_window=max_context_window,
        input_path=input_jsonl,
    )

    submitted_rows = set(row_payloads.keys())
    if not submitted_rows:
        print("没有可提交的行（可能都已续测完成或被上下文窗口过滤）。")
        raise SystemExit(0)

    print(
        f"准备提交行数: {len(submitted_rows)}，"
        f"上下文超限跳过: {len(skipped_context_rows)}，"
        f"CSV 已有结果: {len(tested_rows)}"
    )
    save_json(row_payloads_json, {str(k): v for k, v in row_payloads.items()})

    metadata = {
        "version": 1,
        "needle": needle,
        "model": model_name,
        "run_dir": str(run_dir),
        "csv_path": str(csv_path),
        "input_jsonl": str(input_jsonl),
        "output_jsonl": str(output_jsonl),
        "error_jsonl": str(error_jsonl),
        "row_payloads_json": str(row_payloads_json),
        "input_file_id": None,
        "batch_id": None,
        "batch_status": None,
        "completion_window": str(args.completion_window),
        "poll_interval_seconds": int(args.poll_interval_seconds),
        "submitted_rows": len(submitted_rows),
        "submitted_row_ids": sorted(submitted_rows),
        "skipped_context_rows": sorted(skipped_context_rows),
        "tested_rows": len(tested_rows),
    }
    save_json(meta_json, metadata)
    print(f"prepare 完成，run_dir: {run_dir}")
    return run_dir


def load_meta_or_fail(run_dir: Path) -> tuple[Path, dict[str, Any]]:
    meta_json = run_dir / "meta.json"
    if not meta_json.exists():
        raise FileNotFoundError(f"meta.json 不存在: {meta_json}")
    return meta_json, load_json(meta_json)


def stage_upload(args: argparse.Namespace, run_dir: Path) -> None:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_name = str(metadata["model"])
    input_jsonl = Path(metadata["input_jsonl"])

    if not input_jsonl.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    client = make_client(model_name)
    input_file_id = upload_input_file(client, input_jsonl)

    metadata["input_file_id"] = input_file_id
    metadata["batch_id"] = None
    metadata["batch_status"] = None
    save_json(meta_json, metadata)
    print(f"upload 完成，input_file_id: {input_file_id}")


def stage_create(args: argparse.Namespace, run_dir: Path) -> str:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_name = str(metadata["model"])
    input_file_id = metadata.get("input_file_id")
    if not input_file_id:
        raise ValueError("未找到 input_file_id。请先执行 upload。")

    completion_window = str(args.completion_window or metadata.get("completion_window") or "24h")
    client = make_client(model_name)
    batch_id = create_batch(client, str(input_file_id), completion_window=completion_window)
    metadata["batch_id"] = batch_id
    metadata["batch_status"] = "validating"
    metadata["completion_window"] = completion_window
    save_json(meta_json, metadata)
    print(f"create 完成，batch_id: {batch_id}")
    return batch_id


def stage_wait(args: argparse.Namespace, run_dir: Path) -> Any:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_name = str(metadata["model"])
    batch_id = args.batch_id or metadata.get("batch_id")
    if not batch_id:
        raise ValueError("未找到 batch_id。请先执行 create，或通过 --batch-id 指定。")

    poll_interval = int(args.poll_interval_seconds or metadata.get("poll_interval_seconds") or 10)
    client = make_client(model_name)
    batch = poll_batch(client, str(batch_id), poll_interval_seconds=poll_interval)
    metadata["batch_id"] = str(batch_id)
    metadata["batch_status"] = batch.status
    save_json(meta_json, metadata)
    print(f"wait 完成，最终状态: {batch.status}")
    return batch


def stage_collect(args: argparse.Namespace, run_dir: Path, batch_obj: Any | None = None) -> None:
    meta_json, metadata = load_meta_or_fail(run_dir)
    model_name = str(metadata["model"])
    batch_id = args.batch_id or metadata.get("batch_id")
    if not batch_id:
        raise ValueError("未找到 batch_id。请先执行 submit，或通过 --batch-id 指定。")

    client = make_client(model_name)
    batch = batch_obj if batch_obj is not None else client.batches.retrieve(str(batch_id))
    metadata["batch_id"] = str(batch_id)
    metadata["batch_status"] = batch.status

    if batch.status != "completed":
        save_json(meta_json, metadata)
        raise RuntimeError(f"batch 状态不是 completed，当前为: {batch.status}")

    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        save_json(meta_json, metadata)
        raise RuntimeError("Batch 已完成但 output_file_id 为空")

    output_jsonl = Path(metadata["output_jsonl"])
    error_jsonl = Path(metadata["error_jsonl"])
    csv_path = Path(metadata["csv_path"])
    row_payloads_json = Path(metadata["row_payloads_json"])
    if not row_payloads_json.exists():
        raise FileNotFoundError(f"row_payloads.json 不存在: {row_payloads_json}")

    row_payloads_raw = load_json(row_payloads_json)
    row_payloads = {int(k): v for k, v in row_payloads_raw.items()}

    output_content = client.files.content(output_file_id)
    output_text = extract_text_from_file_content(output_content)
    save_text(output_jsonl, output_text)
    metadata["output_file_id"] = output_file_id

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        error_content = client.files.content(error_file_id)
        error_text = extract_text_from_file_content(error_content)
        save_text(error_jsonl, error_text)
        metadata["error_file_id"] = error_file_id
    else:
        metadata["error_file_id"] = None

    row_results, failed_rows_from_output = parse_batch_output(output_text, row_payloads)
    submitted_rows = set(int(v) for v in metadata.get("submitted_row_ids", []))
    if not submitted_rows:
        submitted_rows = set(row_payloads.keys())
    missing_rows = submitted_rows - set(row_results.keys())
    failed_rows = set(failed_rows_from_output) | set(missing_rows)
    success_rows = sorted(set(row_results.keys()) - failed_rows)

    if failed_rows:
        print(f"[SKIP] 以下行请求失败，不写入 CSV: {sorted(failed_rows)}")

    rows_to_write = [row_results[idx] for idx in success_rows]
    append_results_csv(csv_path, rows_to_write)

    metadata["success_rows"] = success_rows
    metadata["failed_rows"] = sorted(failed_rows)
    metadata["written_rows"] = len(rows_to_write)
    save_json(meta_json, metadata)

    print(f"已写入 CSV: {csv_path}")
    print(f"本次成功写入: {len(rows_to_write)} 行，失败跳过: {len(failed_rows)} 行")
    print(f"中间产物目录: {run_dir}")


def main() -> None:
    args = parse_args()
    step = str(args.step)
    # Backward-compatible aliases
    if step == "submit":
        step = "upload"
    elif step == "poll":
        step = "wait"

    if step in {"upload", "create", "wait", "collect"} and not args.run_dir:
        raise ValueError(f"--step {step} 需要传入 --run-dir")

    if step == "prepare":
        stage_prepare(args)
        return

    if step == "upload":
        stage_upload(args, Path(args.run_dir))
        return

    if step == "create":
        stage_create(args, Path(args.run_dir))
        return

    if step == "wait":
        stage_wait(args, Path(args.run_dir))
        return

    if step == "collect":
        stage_collect(args, Path(args.run_dir))
        return

    run_dir = stage_prepare(args)
    stage_upload(args, run_dir)
    stage_create(args, run_dir)
    batch = stage_wait(args, run_dir)
    stage_collect(args, run_dir, batch_obj=batch)


if __name__ == "__main__":
    main()
