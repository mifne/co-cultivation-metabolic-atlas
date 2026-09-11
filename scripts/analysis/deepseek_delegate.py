"""Reusable helper for delegating large-context (bulk document/code reading)
work to DeepSeek V4 Flash via OpenRouter, so Claude's own context/tokens
are spent only on synthesis and judgment, not on ingesting raw sources.

Why this exists
----------------
Claude (Sonnet 5) is priced at $2.00 / $10.00 per 1M input/output tokens
(Anthropic API pricing, checked via the claude-api skill on 2026-09-11).
DeepSeek V4 Flash (via OpenRouter) is priced at roughly $0.0886 / $0.177
per 1M input/output tokens. That is approximately 23x cheaper on input
and 56x cheaper on output. Any task that requires reading more than a
handful of files, or more than roughly 10-20 KB of raw source/docs,
should be routed through this helper instead of Claude's own Read tool:
DeepSeek does the bulk reading/extraction, Claude only reads its
(much smaller) output and does the critical synthesis.

This module always prints a cost comparison (actual DeepSeek cost vs.
what the same input would have cost Claude directly) so the tradeoff
stays visible, and never silently swallows a truncated response - it
retries once on a non-"stop" finish_reason before giving up.

CLI usage
---------
    OPENROUTER_API_KEY=... python3 scripts/analysis/deepseek_delegate.py \\
        --files "docs/FOO*.md" src/bar.py \\
        --task-file /tmp/task_prompt.txt \\
        --out tmp/foo_extraction.md \\
        --label "foo investigation"

Library usage
-------------
    from deepseek_delegate import delegate

    result = delegate(
        files=["docs/FOO.md", "src/bar.py"],
        task_prompt="...",
        out_path=ROOT / "tmp" / "foo.md",
        label="foo investigation",
    )
    print(result.content)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]

MODEL = "deepseek/deepseek-v4-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Pricing per token (not per million) - see module docstring for source/date.
DEEPSEEK_PROMPT_PRICE = 0.0000000886
DEEPSEEK_COMPLETION_PRICE = 0.000000177212

# Claude prices per 1M tokens (input, output), checked via the claude-api
# skill on 2026-09-11. Used only to show what the same bulk reading would
# have cost if Claude had done it directly.
CLAUDE_PRICING_PER_MTOK = {
    "opus-5": (5.00, 25.00),
    "sonnet-5": (2.00, 10.00),
    "haiku-4-5": (1.00, 5.00),
}
DEFAULT_CLAUDE_MODEL = "opus-5"

DEFAULT_SYSTEM_PROMPT = """\
あなたは計算生物学・ソフトウェア工学に詳しい研究アシスタントです。
大量のドキュメント・生データ・ソースコードを正確に読み、後段の作業者(別のAI)が
そのまま検証・統合できる抽出ノートを作成します。

厳守事項:
- 数値・コード上の事実は必ず原文のとおりに転記すること。丸めたり創作したりしない。
- ファイル内の具体的な行番号は、確信が持てる場合を除き引用しないこと。行番号より
  関数名・変数名・クラス名など安定した識別子を優先して引用すること。
- 出典を必ず [出典: ファイル名] の形式で各項目に付けること。
- 確信が持てない推測・一般化・引用(論文名やDOIなど)を作らないこと。分からない場合は
  「確信が持てない」「原文に記載なし」と明記すること。
- 日本語で出力すること。
"""


@dataclass
class DelegateResult:
    content: str
    usage: dict
    finish_reason: str | None
    native_finish_reason: str | None
    out_path: Path | None
    input_chars: int


def _resolve_files(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        candidate = Path(pattern)
        if not candidate.is_absolute():
            candidate = ROOT / pattern
        matches = sorted(Path(p) for p in glob.glob(str(candidate)))
        if not matches and candidate.exists():
            matches = [candidate]
        if not matches:
            print(f"WARNING: no match for {pattern}", file=sys.stderr)
        paths.extend(matches)
    return paths


def _read_files(paths: list[Path]) -> str:
    parts = []
    for path in paths:
        rel = path.relative_to(ROOT) if path.is_relative_to(ROOT) else path
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except IsADirectoryError:
            continue
        fence = "```python\n" if path.suffix == ".py" else ""
        fence_close = "\n```" if fence else ""
        parts.append(f"\n\n===== FILE: {rel} =====\n{fence}{text}{fence_close}\n")
    return "".join(parts)


def _call_openrouter(payload: dict, api_key: str, timeout: int) -> dict:
    resp = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        data=json.dumps(payload),
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"OpenRouter error: {data['error']}")
    return data


def delegate(
    files: list[str],
    task_prompt: str,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    out_path: Path | str | None = None,
    label: str = "task",
    model: str = MODEL,
    max_tokens: int = 16000,
    temperature: float = 0.1,
    timeout: int = 900,
    max_retries: int = 2,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
) -> DelegateResult:
    """Send `files` + `task_prompt` to DeepSeek V4 Flash and return the result.

    Retries once (by default) if the response looks truncated (finish_reason
    is not "stop"), since OpenRouter provider routing occasionally returns a
    short/incomplete response for large prompts.
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in the environment")

    paths = _resolve_files(files)
    if not paths:
        raise SystemExit("No input files resolved; nothing to send to DeepSeek")

    file_text = _read_files(paths)
    user_content = f"{task_prompt}\n\n{file_text}"
    input_chars = len(system_prompt) + len(user_content)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    print(
        f"[{label}] sending {len(paths)} file(s), {input_chars} chars "
        f"(~{input_chars // 3} tokens est.) to {model}"
    )

    data = None
    finish_reason = None
    for attempt in range(1, max_retries + 2):
        data = _call_openrouter(payload, api_key, timeout)
        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        content_len = len(choice["message"]["content"])
        if finish_reason == "stop" or attempt > max_retries:
            break
        print(
            f"[{label}] attempt {attempt}: finish_reason={finish_reason}, "
            f"content only {content_len} chars - retrying",
            file=sys.stderr,
        )
        time.sleep(2 * attempt)

    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    native_finish_reason = choice.get("native_finish_reason")

    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    deepseek_cost = usage.get("cost")
    if deepseek_cost is None:
        deepseek_cost = (
            prompt_tokens * DEEPSEEK_PROMPT_PRICE
            + completion_tokens * DEEPSEEK_COMPLETION_PRICE
        )
    claude_in, claude_out = CLAUDE_PRICING_PER_MTOK.get(
        claude_model, CLAUDE_PRICING_PER_MTOK[DEFAULT_CLAUDE_MODEL]
    )
    claude_equivalent_cost = (
        prompt_tokens * claude_in / 1_000_000 + completion_tokens * claude_out / 1_000_000
    )
    savings = claude_equivalent_cost - deepseek_cost
    multiplier = (claude_equivalent_cost / deepseek_cost) if deepseek_cost else float("inf")

    print(
        f"[{label}] tokens: prompt={prompt_tokens} completion={completion_tokens} | "
        f"DeepSeek cost=${deepseek_cost:.5f} | "
        f"Claude {claude_model} equivalent=${claude_equivalent_cost:.5f} | "
        f"saved=${savings:.5f} ({multiplier:.1f}x cheaper)"
    )

    out_path_resolved = None
    if out_path is not None:
        out_path_resolved = Path(out_path)
        if not out_path_resolved.is_absolute():
            out_path_resolved = ROOT / out_path_resolved
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"<!-- generated {datetime.now(timezone.utc).isoformat()} label={label} "
            f"model={model} usage={json.dumps(usage)} "
            f"finish_reason={finish_reason} native_finish_reason={native_finish_reason} "
            f"deepseek_cost={deepseek_cost:.5f} "
            f"claude_{claude_model}_equivalent_cost={claude_equivalent_cost:.5f} -->\n\n"
        )
        out_path_resolved.write_text(header + content, encoding="utf-8")
        print(f"[{label}] wrote {out_path_resolved} ({len(content)} chars)")

    return DelegateResult(
        content=content,
        usage=usage,
        finish_reason=finish_reason,
        native_finish_reason=native_finish_reason,
        out_path=out_path_resolved,
        input_chars=input_chars,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", nargs="+", required=True, help="File paths or glob patterns")
    parser.add_argument("--task-file", help="Path to a text file containing the task prompt")
    parser.add_argument("--task", help="Inline task prompt (alternative to --task-file)")
    parser.add_argument("--out", required=True, help="Output path for the extraction")
    parser.add_argument("--label", default="task", help="Short label for logging")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-tokens", type=int, default=16000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--claude-model",
        default=DEFAULT_CLAUDE_MODEL,
        choices=sorted(CLAUDE_PRICING_PER_MTOK),
        help="Which Claude model to price the comparison against (the session's model)",
    )
    args = parser.parse_args()

    if args.task_file:
        task_prompt = Path(args.task_file).read_text(encoding="utf-8")
    elif args.task:
        task_prompt = args.task
    else:
        raise SystemExit("Provide --task-file or --task")

    delegate(
        files=args.files,
        task_prompt=task_prompt,
        out_path=args.out,
        label=args.label,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        claude_model=args.claude_model,
    )


if __name__ == "__main__":
    main()
