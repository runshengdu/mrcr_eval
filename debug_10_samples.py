import asyncio
import json
import re

from main import (
    dataset,
    grade,
    n_tokens,
    client,
    MODEL,
    MAX_CONTEXT_WINDOW,
)


def _normalize_leading_local(s: str) -> str:
    return re.sub(r"^[\ufeff\u200b\u200c\u200d\s]+", "", s.replace("\r\n", "\n"))

async def debug_single_row(idx, row) -> bool:
    """Call the model once for a single row and print detailed info.

    Returns True if the call succeeded and grade was computed,
    otherwise False (e.g., skipped or error).
    """
    messages = json.loads(row["prompt"])
    token_count = n_tokens(messages)

    if token_count > MAX_CONTEXT_WINDOW:
        print(
            f"[SKIP] row={idx} token_count={token_count} exceeds MAX_CONTEXT_WINDOW={MAX_CONTEXT_WINDOW}",
            flush=True,
        )
        return False

    try:
        completion = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            timeout=30.0,
            extra_body={"reasoning": {"enabled": True}},
        )
        response = completion.choices[0].message.content

        # Mirror grade()'s preprocessing so we can see exactly what is matched
        response_no_think = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        response_norm = _normalize_leading_local(response_no_think)
        answer_norm = _normalize_leading_local(row["answer"])
        prefix = row["random_string_to_prepend"]

        g = grade(response, row["answer"], prefix)
        if g == 0.0:
            print("--- DEBUG (grade == 0) ---", flush=True)
            print("prefix:", prefix, flush=True)
            print("response_norm head:", response_norm[:200], flush=True)
            print("answer_norm head:", answer_norm[:200], flush=True)
            print(
                "response_norm.startswith(prefix):",
                response_norm.startswith(prefix),
                flush=True,
            )
            print("---------------------------", flush=True)
        else:
            print(f"[SUCCESS] row={idx} grade={g:.6f}", flush=True)
        return True
    except Exception as e:
        print(f"[ERROR] row={idx} exception: {e}", flush=True)
        return False


async def main() -> None:
    # Shuffle the whole dataset and take the first 10 that run successfully
    shuffled = dataset.sample(frac=1.0, random_state=None)

    success = 0
    for idx, row in shuffled.iterrows():
        ok = await debug_single_row(idx, row)
        if ok:
            success += 1
        if success >= 10:
            break

    print(f"Debug finished. Successful samples: {success}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
