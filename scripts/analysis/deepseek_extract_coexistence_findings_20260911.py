"""Send the docs relevant to 3-species (OR16/NS21/P. freudenreichii) coexistence
to DeepSeek V4 Flash via OpenRouter and save its extraction for Claude to
synthesize into a Word report.

Usage:
    OPENROUTER_API_KEY=... python3 scripts/analysis/deepseek_extract_coexistence_findings_20260911.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "tmp"
OUT_DIR.mkdir(exist_ok=True)
OUT_PATH = OUT_DIR / "deepseek_coexistence_extraction_20260911.md"

MODEL = "deepseek/deepseek-v4-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

SOURCE_DOCS = [
    "docs/PFREUDENREICHII_VALUE_AUDIT_20260908.md",
    "docs/PFREUDENREICHII_CAUSAL_AUDIT_20260908.md",
    "docs/AUDITED_SYMBIOSIS_INTERPRETATION_20260908.md",
    "docs/CULTIVATION_MODEL_AUDIT_20260908.md",
    "docs/CULTIVATION_MODEL_REPAIR_20260908.md",
    "docs/PHYSIOLOGY_IMPLEMENTATION_20260908.md",
    "docs/PHYSIOLOGY_LONG_VALIDATION_20260908.md",
    "docs/RESOLVED_DYNAMICS_20260908.md",
    "docs/EQUAL_BUDGET_COMPARISON_20260908.md",
    "docs/OXYGEN_SCHEDULE_THIRD_SPECIES_20260910.md",
    "docs/WCFS1_COEXISTENCE_OPTIMIZATION.md",
    "docs/THREE_PUMP_FEED_STRATEGY.md",
    "docs/RL_FEDBATCH_COCULTURE_MILESTONES.md",
    "docs/ONE_L_JAR_FERMENTER_CONTROL_DESIGN.md",
    "docs/ONE_L_JAR_SMOKE_REPORT.md",
    "docs/NS21_PHA_MODEL_CURATION.md",
]

SYSTEM_PROMPT = """\
あなたは計算生物学の研究アシスタントです。天然ゴム分解・PHA変換の
微生物コンソーシアム(OR16, NS21, P. freudenreichii等)に関する複数の
研究ドキュメントを読み、後段の作業者(別のAI)が査読可能なWord報告書を
作成するための「抽出ノート」を作成してください。

厳守事項:
- 数値は必ず原文の値をそのまま転記すること。丸めたり創作したりしない。
- 各ドキュメントが明記している「限界・未検証事項・主張しないこと」は省略せず、
  該当する発見のすぐ近くに書くこと。
- ドキュメント間で数値や結論が食い違う場合は、その食い違いを明示すること。
- 出典を必ず [出典: ファイル名] の形式で各項目に付けること。
- 憶測や一般化(例:「本モデルの外挿として一般に成り立つ」等)を追加しないこと。
  ドキュメントに書かれている範囲を超えて解釈を広げない。
- 日本語で出力すること。
"""

USER_INSTRUCTION = """\
以下は複数の研究ドキュメント(Markdown)です。焦点は「OR16 + NS21 の2種培養に
P. freudenreichii を加えた3種共存系が何をもたらすか」という問いです。

各ドキュメントについて、以下の構成で抽出してください:

## [ファイル名]
- 目的・実験条件の要約(1-3行)
- 3種共存に関する主要な数値結果(表があれば表のまま転記)
- 明記されている限界・未検証事項・「主張しないこと」
- 他のドキュメントと関連しそうな箇所(あれば)

すべてのドキュメントの抽出が終わったら、最後に「## 横断的な矛盾・注意点」として
ドキュメント間で数値や結論が食い違っている箇所、または解釈に注意が必要な箇所を
箇条書きでまとめてください。創作は禁止し、原文にない結論は書かないでください。

---

"""


def build_payload() -> dict:
    parts = [USER_INSTRUCTION]
    for rel in SOURCE_DOCS:
        path = ROOT / rel
        if not path.exists():
            print(f"WARNING: missing {rel}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        parts.append(f"\n\n===== FILE: {rel} =====\n{text}\n")
    user_content = "".join(parts)
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.1,
        "max_tokens": 16000,
    }


def main() -> None:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set in the environment")

    payload = build_payload()
    approx_chars = sum(len(m["content"]) for m in payload["messages"])
    print(f"Sending request to {MODEL} ({approx_chars} chars ~ {approx_chars // 3} tokens est.)")

    resp = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(payload),
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise SystemExit(f"OpenRouter error: {data['error']}")

    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})

    header = (
        f"<!-- generated {datetime.now(timezone.utc).isoformat()} "
        f"model={MODEL} usage={json.dumps(usage)} -->\n\n"
    )
    OUT_PATH.write_text(header + content, encoding="utf-8")
    print(f"Wrote extraction to {OUT_PATH} ({len(content)} chars)")
    print(f"Usage: {usage}")


if __name__ == "__main__":
    main()
