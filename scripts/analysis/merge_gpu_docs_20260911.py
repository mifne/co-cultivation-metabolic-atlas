"""Drive the actual GPU-docs consolidation merges via DeepSeek V4 Flash.

Reads tmp/gpu_groups.json (15 groups covering the 57 docs/GPU_*.md-family
files) and, for each group, asks DeepSeek to produce one consolidated
Markdown document from the full text of its source files. Output goes to
tmp/gpu_merge_output/<name>.md for review before being moved into docs/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from deepseek_delegate import delegate  # noqa: E402

GROUPS_PATH = ROOT / "tmp" / "gpu_groups.json"
OUT_DIR = ROOT / "tmp" / "gpu_merge_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = """\
あなたは計算科学の技術文書エディタです。GPU加速dFBA/LPソルバー開発の複数の
関連ドキュメント(多くは「計画」→「結果」のペア、または同一テーマの複数回の
試行記録)を渡すので、それらを1つの整理された統合ドキュメントへ書き直します。

厳守事項:
- 数値・結論は原文のとおりに転記すること。丸めたり創作したりしない。
- 各セクションの内容がどの元ファイルに由来するかを見出しまたは注記で示すこと
  (例:「(出典: GPU_HYBRID_METHODS_20260904.md, 2026-09-04)」)。
- 元ファイル間で数値や結論が食い違う場合、それを統合時に黙って上書きせず、
  「食い違いに関する注記」として明示すること。
- 「未達」「未実装」「設計案のみ」等の限界・ステータスは省略しないこと。
- 時系列(日付)がある場合は時系列順に並べ、最新の結論を末尾または冒頭の要約に
  明記すること。
- 単なる切り貼りではなく、重複する記述(同じ数値・同じ前提の繰り返し)は整理して
  1回にまとめ、読みやすくすること。ただし情報を削ってはならない。
- 出力は元ドキュメントの主要な言語(日本語または英語)に合わせること。英語主体の
  資料が中心なら英語で、日本語主体なら日本語で出力する。
- Markdown形式で、末尾に「## 構成元ファイル」として元ファイル名と元の日付を
  列挙すること。
"""


def build_task_prompt(group_name: str, files: list[str]) -> str:
    return (
        f"以下は統合先ファイル `{group_name}.md` を構成する{len(files)}件の"
        "ドキュメントです。上記の方針に従って1つの統合ドキュメントに書き直して"
        "ください。冒頭に統合ドキュメント全体のタイトルと、内容を2-3行で要約した"
        "導入を置いてください。\n\n"
    )


def main() -> None:
    data = json.loads(GROUPS_PATH.read_text(encoding="utf-8"))
    groups = data["groups"]

    only = sys.argv[1:] if len(sys.argv) > 1 else None

    for g in groups:
        name = g["name"]
        if only and name not in only:
            continue
        files = [f"docs/{f}" for f in g["files"]]
        out_path = OUT_DIR / f"{name}.md"
        task_prompt = build_task_prompt(name, g["files"])
        result = delegate(
            files=files,
            task_prompt=task_prompt,
            system_prompt=SYSTEM_PROMPT,
            out_path=out_path,
            label=name,
            claude_model="sonnet-5",
            max_tokens=24000,
        )
        if result.finish_reason != "stop":
            print(f"WARNING: {name} finished with reason={result.finish_reason}", file=sys.stderr)


if __name__ == "__main__":
    main()
