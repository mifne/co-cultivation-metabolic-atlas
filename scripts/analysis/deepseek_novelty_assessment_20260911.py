"""Broader DeepSeek V4 Flash pass: surface anything missed in the first
extraction, sanity-check the dFBA methodology against the actual source
code, and assess how scientifically interesting/novel the third-species
(P. freudenreichii) coexistence finding is.

All heavy document/code reading happens inside this script (sent to
DeepSeek); Claude only reads the resulting extraction file, not the raw
sources, to keep this large-context work off Claude's context per the
user's instruction.

Usage:
    OPENROUTER_API_KEY=... python3 scripts/analysis/deepseek_novelty_assessment_20260911.py
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
OUT_PATH = OUT_DIR / "deepseek_novelty_assessment_20260911.md"

MODEL = "deepseek/deepseek-v4-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Docs already digested in the first pass (context, not re-explained here)
PRIOR_DOCS = [
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

# New material for this pass: broader doc net + raw numeric result reports
# + the actual simulator source code, so DeepSeek can sanity-check the
# biology/physics against the implementation, not just the prose summaries.
NEW_DOCS = [
    "docs/NEXT_PHASE_PLAN_AFTER_RL_EFFECT.md",
    "docs/RL_EFFECT_4096_REPORT.md",
    "docs/FEED_VALIDATION_EXPERIMENT.md",
    "docs/RL_PRETRAINING_FEASIBILITY_AUDIT.md",
    "docs/GENE_ID_STANDARD.md",
]

RESULT_REPORTS = [
    "results/oxygen_schedule_extension_20260910/REPORT_JA.md",
    "results/oxygen_schedule_extension_20260910/summary.json",
    "results/oxygen_schedule_closure_20260910/REPORT_JA.md",
    "results/oxygen_schedule_closure_20260910/summary.json",
    "results/pfreud_causal_20260908/REPORT_JA.md",
    "results/pfreud_causal_20260908/COLLECTION_STATUS_NOTE_JA.md",
    "results/third_species_value_20260910/summary.json",
]

SOURCE_FILES = [
    "src/physiology_dfba.py",
    "src/cultivation_numerics.py",
    "src/resolved_dfba.py",
    "src/audited_dfba.py",
    "src/b12_evidence.py",
]

SYSTEM_PROMPT = """\
あなたは計算生物学・微生物生態学に詳しい研究アシスタントです。
天然ゴム分解菌 OR16、NS21 に、プロピオン酸産生菌 P. freudenreichii (Pf) を
第3の菌種として加えた3種コンソーシアムに関する、dFBA(動的フラックスバランス
解析)シミュレーション研究群を精査します。

すでに一次抽出(前回パス)は完了しており、以下の主要な結論が確認済みです:
- 3種優位は酸素供給条件に強く依存し、非単調(kLa中間域や曝気配分の悪いパルスで
  有利、kLa極端に低い/高いで不利)。
- 時間刻みを粗くすると優位が過大評価される(1時間刻みで+20%、秒刻みで逆転)。
- Pfの価値は「PHA上限を上げる」のではなく「悪い曝気配分の落ち込みを緩和する」。
- 3HV上昇はPf特有の効果ではない(2種+プロピオン酸直接投与でも上昇)。
- 維持代謝・死滅率・B12依存・kLa・Pf酸素表現型など多数のパラメータが未校正。

今回は以下の3つの作業を行ってください。

## 作業1: 見落としの確認
新たに渡す追加ドキュメント・生データ(results配下のsummary.json、REPORT_JA.md)・
シミュレータのソースコード(src/*.py)を読み、前回の抽出に含まれていない、
3種共存の解釈に影響しうる情報があれば指摘してください(数値・実装上の仮定・
未校正パラメータなど)。特に、シミュレータのソースコードが実際に実装している
生物学的仮定(酸素分配式、窒素配分式、PHA蓄積上限の扱い、B12授受の有無など)を
確認し、ドキュメントの記述と食い違いがないか照合してください。なければ「なし」
と明記してください。

## 作業2: 手法的健全性の評価
dFBAによる合成コンソーシアム設計として、この一連の研究の手法は標準的な実践と
比べてどうか評価してください。強み(例: 時間刻み収束を必ず確認する、SHA256で
ソース一致を照合する、決定論的比較であることを明記する、否定的結果も隠さない)
と、弱み・欠落(例: 反復・統計検定がない、多数の未校正パラメータ、B12授受経路が
モデルにない、酸素分配則が単純化されている等)を、具体的な実装箇所を引用しながら
列挙してください。

## 作業3: 科学的な面白さ・新規性の評価
あなたの一般知識(学習データの範囲内、Web検索なし)に基づき、以下の観点で
この研究がどれくらい「面白い・新規性がある」データかを評価してください:
(a) 天然ゴム分解菌コンソーシアムへのPHA/3HV組成制御という応用の位置づけ。
(b) プロピオン酸/奇数鎖前駆体のクロスフィーディングによるPHBV組成制御という
    手法自体の、代謝工学・合成生物学における一般的な位置づけ。
(c) 「共生関係の価値が資源供給条件(酸素)に対して非単調に変化する」という
    現象が、微生物生態学の既存理論(資源比理論、ストレス勾配仮説、競争と協力の
    条件依存的転換など)とどう関連しうるか。
(d) 計算駆動(dFBA)で先に条件依存性を発見してから湿式実験へ進む、という
    研究の進め方自体の位置づけ。

**厳守事項**: 具体的な論文名・著者名・DOIなど、確信が持てない引用は絶対に
作らないこと(ハルシネーション厳禁)。一般的な理論名・概念名(例:資源比理論)を
挙げるのは良いが、「〇〇年の△△論文で示された」のような具体的すぎる主張は、
確信がない限り避け、「一般的にこのような現象は◯◯として知られる」という
抽象度に留めること。分からない・確信が持てない場合は「確信が持てない」と
明記すること。

最後に「総合評価」として、この結果が研究として面白い理由・面白くない理由を
バランスよく整理し、次に何を示せれば説得力が増すかを述べてください。
過大評価も過小評価もせず、ドキュメント自身が明記する限界を尊重してください。
日本語で出力してください。
"""


def read_or_warn(rel: str) -> str | None:
    path = ROOT / rel
    if not path.exists():
        print(f"WARNING: missing {rel}", file=sys.stderr)
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def build_payload() -> dict:
    parts = ["# 前回抽出済みドキュメント(参考・再確認用)\n"]
    for rel in PRIOR_DOCS:
        text = read_or_warn(rel)
        if text is not None:
            parts.append(f"\n\n===== FILE: {rel} =====\n{text}\n")

    parts.append("\n\n# 追加ドキュメント(未読)\n")
    for rel in NEW_DOCS:
        text = read_or_warn(rel)
        if text is not None:
            parts.append(f"\n\n===== FILE: {rel} =====\n{text}\n")

    parts.append("\n\n# 生データ・実行レポート(未読)\n")
    for rel in RESULT_REPORTS:
        text = read_or_warn(rel)
        if text is not None:
            parts.append(f"\n\n===== FILE: {rel} =====\n{text}\n")

    parts.append("\n\n# シミュレータのソースコード(未読)\n")
    for rel in SOURCE_FILES:
        text = read_or_warn(rel)
        if text is not None:
            parts.append(f"\n\n===== FILE: {rel} =====\n```python\n{text}\n```\n")

    user_content = "".join(parts)
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.2,
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
        timeout=900,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise SystemExit(f"OpenRouter error: {data['error']}")

    (OUT_DIR / "deepseek_novelty_assessment_20260911.raw.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    choice = data["choices"][0]
    content = choice["message"]["content"]
    usage = data.get("usage", {})
    finish_reason = choice.get("finish_reason")
    native_finish_reason = choice.get("native_finish_reason")

    header = (
        f"<!-- generated {datetime.now(timezone.utc).isoformat()} "
        f"model={MODEL} usage={json.dumps(usage)} "
        f"finish_reason={finish_reason} native_finish_reason={native_finish_reason} -->\n\n"
    )
    OUT_PATH.write_text(header + content, encoding="utf-8")
    print(f"Wrote extraction to {OUT_PATH} ({len(content)} chars)")
    print(f"Usage: {usage}")
    print(f"finish_reason={finish_reason} native_finish_reason={native_finish_reason}")


if __name__ == "__main__":
    main()
